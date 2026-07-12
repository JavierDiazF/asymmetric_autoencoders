"""
Same architecture sweep as compression_sweep.py, but built entirely on
pythae (https://github.com/clementchadebec/benchmark_VAE) -- no dependency
on this repo's own encoder.py / decoder.py / autoencoder.py / train.py.
Everything AE-related (model, encoder/decoder base classes, config, training
loop) comes from the pythae package; only the sweep grids, the synthetic
data generator, the MAC/param/energy bookkeeping and the plot styling are
local (they're generic utilities, not an AE framework).

Model: pythae.models.AE, configured via pythae.models.AEConfig.
Encoder/decoder: pythae ships a *fixed* MLP (Encoder_AE_MLP / Decoder_AE_MLP
-- one 512-unit hidden layer, see pythae/models/nn/default_architectures.py)
with no depth/width parameters, so it can't run a depth sweep out of the box.
MLPEncoder/MLPDecoder below fill that gap: they subclass pythae's own
BaseEncoder/BaseDecoder and follow the exact same convention pythae's default
MLP uses internally (ModuleList of Linear+activation blocks, a Sigmoid
output layer since data is min-max scaled to [0, 1], ModelOutput(embedding=..)
/ ModelOutput(reconstruction=..)) -- just with a configurable hidden_dims
list instead of a hardcoded single layer.
Training: pythae.pipelines.TrainingPipeline + pythae.trainers.BaseTrainerConfig
-- pythae's own documented end-to-end training entry point. BaseTrainerConfig
has no early-stopping field, so each config trains for a fixed epoch budget
(same budgets as compression_sweep.py); the checkpoint used for evaluation is
whichever epoch had the best eval loss, tracked internally by BaseTrainer as
`trainer._best_model` -- not a local re-implementation of early stopping.
BaseTrainer always writes a "final_model" checkpoint to `output_dir`; each
config trains inside its own `tempfile.TemporaryDirectory` so the sweep
doesn't leave dozens of checkpoint folders behind.

Whether pythae supports asymmetric encoder/decoder depth: yes.
pythae.models.AE takes independent `encoder=` / `decoder=` instances (any
BaseEncoder/BaseDecoder subclass) with no constraint that they share depth or
width -- only that encoder.embedding and decoder's expected input agree on
latent_dim. Sweep C below builds exactly that: encoder/decoder hidden-layer
counts varied independently, e.g. (1, 3) = 1-hidden-layer encoder / 3-hidden-
layer decoder.

Three sweeps (A and B mirror compression_sweep.py; C is new):
  A) DIMENSION   -- input_dim x compression ratio (latent_dim), fixed depth.
  B) DEPTH       -- symmetric hidden-layer count, encoder == decoder, on
     linear and nonlinear data.
  C) ASYMMETRY   -- fixed (input_dim, latent_dim), encoder/decoder hidden
     layers varied independently: (1,1), (3,3), (1,3), (3,1).

Outputs (in results/):
  pythae_compression_sweep.csv  -- one row per trained configuration (A+B+C)
  pythae_compression_sweep.png  -- sweep A + B figure
  pythae_asymmetric_sweep.png   -- sweep C figure
"""

import csv
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from pythae.models import AE, AEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

torch.manual_seed(42)
rng = np.random.default_rng(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config — same grids/budgets as compression_sweep.py, so the two are a
# like-for-like comparison ────────────────────────────────────────────────

INPUT_DIMS              = [32, 64, 128, 256, 512, 1024]     # Sweep A
COMPRESSION_RATIOS      = [2, 4, 8, 16, 32]        # latent_dim = input_dim // ratio
HIDDEN_LAYERS_DIM_SWEEP = 2                        # depth held fixed during Sweep A

DEPTH_SWEEP_INPUT_DIM  = 256                       # Sweep B / C (fixed dims)
DEPTH_SWEEP_LATENT_DIM = 32                        # ratio 8, safely above TRUE_RANK
HIDDEN_LAYER_COUNTS    = [0, 1, 2, 3, 4]
DEPTH_SWEEP_EPOCHS     = 300                        # no early stopping -> fixed budget

ASYM_INPUT_DIM  = DEPTH_SWEEP_INPUT_DIM
ASYM_LATENT_DIM = DEPTH_SWEEP_LATENT_DIM
ASYM_CONFIGS: List[Tuple[int, int]] = [(1, 1), (3, 3), (1, 3), (3, 1)]

TRUE_RANK  = 12        # intrinsic dimensionality of the synthetic data
NOISE_STD  = 0.05       # additive observation noise (absolute, on ~unit-variance features)
NONLINEAR_HIDDEN = 24   # width of the fixed random generator used for the nonlinear variant

ACTIVATION = "elu"
EPOCHS     = 60
N_TRAIN, N_VAL, N_TEST = 8_000, 1_000, 2_000

ENERGY_PJ_PER_MAC = 4.6   # Horowitz ISSCC'14, 45nm float32 mult+add -- compute-only estimate

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "pythae_compression_sweep.csv"
PLOT_PATH = RESULTS_DIR / "pythae_compression_sweep.png"
ASYM_PLOT_PATH = RESULTS_DIR / "pythae_asymmetric_sweep.png"

FIELDNAMES = [
    "sweep", "input_dim", "latent_dim",
    "encoder_hidden_layers", "decoder_hidden_layers", "compression_ratio",
    "mse", "n_params", "macs", "encoder_macs", "decoder_macs",
    "energy_est_nJ", "latency_ms",
]


# ── Synthetic low-rank data (identical generator to compression_sweep.py) ──

def build_dataset_for_input_dim(
    input_dim: int, nonlinear: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X = Z @ A + noise (linear) or tanh(Z @ W1) @ W2 + noise (nonlinear),
    Z ~ N(0, I_TRUE_RANK) -- see compression_sweep.py for the full rationale.
    """
    if nonlinear:
        W1 = rng.standard_normal((TRUE_RANK, NONLINEAR_HIDDEN)).astype(np.float32)
        W2 = rng.standard_normal((NONLINEAR_HIDDEN, input_dim)).astype(np.float32)
        W2 /= np.linalg.norm(W2, axis=0, keepdims=True)

        def sample(n: int) -> np.ndarray:
            Z = rng.standard_normal((n, TRUE_RANK)).astype(np.float32)
            X = np.tanh(Z @ W1) @ W2
            X += rng.normal(scale=NOISE_STD, size=X.shape).astype(np.float32)
            return X.astype(np.float32)
    else:
        A = rng.standard_normal((TRUE_RANK, input_dim)).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)

        def sample(n: int) -> np.ndarray:
            Z = rng.standard_normal((n, TRUE_RANK)).astype(np.float32)
            X = Z @ A
            X += rng.normal(scale=NOISE_STD, size=X.shape).astype(np.float32)
            return X.astype(np.float32)

    raw_train, raw_val, raw_test = sample(N_TRAIN), sample(N_VAL), sample(N_TEST)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(raw_train)
    X_val = scaler.transform(raw_val)
    X_test = scaler.transform(raw_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
    )


# ── pythae encoder / decoder ────────────────────────────────────────────────
# Same convention as pythae's own Encoder_AE_MLP / Decoder_AE_MLP
# (models/nn/default_architectures.py), just with a configurable hidden_dims
# list instead of a hardcoded single 512-unit layer.

_ACTIVATIONS = {
    "relu": nn.ReLU, "elu": nn.ELU, "selu": nn.SELU, "gelu": nn.GELU,
    "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "none": nn.Identity,
}


def _hidden_stack(dims: List[int], activation: str) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    blocks: List[nn.Module] = []
    for i in range(len(dims) - 1):
        blocks += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
    return nn.Sequential(*blocks)  # empty hidden_dims -> identity (direct projection)


def interpolate_widths(start: int, end: int, n: int) -> List[int]:
    """n geometrically-spaced interior widths between start and end (both
    exclusive). n=0 -> [] (direct start->end projection)."""
    if n <= 0:
        return []
    sizes = np.geomspace(start, end, num=n + 2)[1:-1]
    return [max(1, int(round(s))) for s in sizes]


class MLPEncoder(BaseEncoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], activation: str = ACTIVATION):
        BaseEncoder.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        dims = [input_dim] + list(hidden_dims)
        self.hidden = _hidden_stack(dims, activation)
        self.embedding = nn.Linear(dims[-1], latent_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        return ModelOutput(embedding=self.embedding(self.hidden(x)))


class MLPDecoder(BaseDecoder):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int], activation: str = ACTIVATION):
        BaseDecoder.__init__(self)
        self.input_dim = output_dim  # matches pythae's own Decoder_AE_MLP attribute name
        dims = [latent_dim] + list(hidden_dims)
        self.hidden = _hidden_stack(dims, activation)
        # pythae's own Decoder_AE_MLP ends on Sigmoid -- data is min-max scaled to [0, 1]
        self.reconstruction_layer = nn.Sequential(nn.Linear(dims[-1], output_dim), nn.Sigmoid())

    def forward(self, z: torch.Tensor, **kwargs) -> ModelOutput:
        return ModelOutput(reconstruction=self.reconstruction_layer(self.hidden(z)))


# ── Cost / energy helpers (generic nn.Module introspection) ────────────────

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_macs(module: nn.Module) -> int:
    """MAC count = sum(in_features * out_features) over every nn.Linear layer."""
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


def measure_latency_ms(model: AE, input_dim: int, device: torch.device,
                        batch_size: int = 1, n_warmup: int = 20, n_repeats: int = 200) -> float:
    """Empirical wall-clock forward-pass latency (encode+decode), averaged over
    n_repeats after a warmup period."""
    model.eval()
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.decoder(model.encoder(x).embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            model.decoder(model.encoder(x).embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_repeats * 1000.0


# ── Training (pythae's own TrainingPipeline / BaseTrainer) ──────────────────

def train_one_config(
    X_train: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor,
    input_dim: int, latent_dim: int, encoder_layers: int, decoder_layers: int,
    epochs: int, batch_size: int = 128,
) -> Tuple[AE, float]:
    encoder = MLPEncoder(input_dim, latent_dim, interpolate_widths(input_dim, latent_dim, encoder_layers))
    decoder = MLPDecoder(latent_dim, input_dim, interpolate_widths(latent_dim, input_dim, decoder_layers))
    model = AE(AEConfig(input_dim=(input_dim,), latent_dim=latent_dim), encoder=encoder, decoder=decoder)

    with tempfile.TemporaryDirectory(prefix="pythae_sweep_") as tmp_dir:
        training_config = BaseTrainerConfig(
            output_dir=tmp_dir,
            num_epochs=epochs,
            learning_rate=2e-3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=256,
            optimizer_cls="Adam",
            steps_saving=None,
            steps_predict=None,
            no_cuda=(DEVICE.type != "cuda"),
        )
        pipeline = TrainingPipeline(model=model, training_config=training_config)

        logging.disable(logging.INFO)  # BaseTrainer logs every epoch at INFO level
        try:
            pipeline(train_data=X_train, eval_data=X_val)
        finally:
            logging.disable(logging.NOTSET)

        trained_model = pipeline.trainer._best_model.to(DEVICE)

    trained_model.eval()
    with torch.no_grad():
        z = trained_model.encoder(X_test.to(DEVICE)).embedding
        x_hat = trained_model.decoder(z).reconstruction
        test_mse = F.mse_loss(x_hat, X_test.to(DEVICE)).item()

    return trained_model, test_mse


def evaluate_model(model: AE, input_dim: int, mse: float) -> Dict[str, float]:
    macs = count_macs(model)
    return {
        "mse": mse,
        "n_params": count_params(model),
        "macs": macs,
        "encoder_macs": count_macs(model.encoder),
        "decoder_macs": count_macs(model.decoder),
        "energy_est_nJ": macs * ENERGY_PJ_PER_MAC / 1000.0,
        "latency_ms": measure_latency_ms(model, input_dim, DEVICE),
    }


# ── sweeps ──────────────────────────────────────────────────────────────────

def run_sweep_a() -> List[dict]:
    """input_dim x compression ratio, fixed symmetric depth."""
    print("=" * 70)
    print(f"[pythae] SWEEP A — dimension  (hidden_layers fixed at {HIDDEN_LAYERS_DIM_SWEEP})")
    print("=" * 70)

    out: List[dict] = []
    for input_dim in INPUT_DIMS:
        X_train, X_val, X_test = build_dataset_for_input_dim(input_dim)

        for ratio in COMPRESSION_RATIOS:
            latent_dim = max(1, input_dim // ratio)
            model, mse = train_one_config(
                X_train, X_val, X_test, input_dim, latent_dim,
                HIDDEN_LAYERS_DIM_SWEEP, HIDDEN_LAYERS_DIM_SWEEP, epochs=EPOCHS,
            )
            metrics = evaluate_model(model, input_dim, mse)

            out.append({
                "sweep": "dimension", "input_dim": input_dim, "latent_dim": latent_dim,
                "encoder_hidden_layers": HIDDEN_LAYERS_DIM_SWEEP,
                "decoder_hidden_layers": HIDDEN_LAYERS_DIM_SWEEP,
                "compression_ratio": input_dim / latent_dim, **metrics,
            })
            print(f"  input_dim={input_dim:>4}  ratio={ratio:>3}  latent_dim={latent_dim:>4}  "
                  f"MSE={mse:.6f}  params={metrics['n_params']:,}  MACs={metrics['macs']:,}")
    return out


def run_sweep_b() -> List[dict]:
    """symmetric hidden-layer count, fixed input_dim/latent_dim, linear vs nonlinear data."""
    out: List[dict] = []

    for nonlinear in (False, True):
        tag = "depth_nonlinear" if nonlinear else "depth_linear"
        print("\n" + "=" * 70)
        print(f"[pythae] SWEEP B — {tag}  (input_dim={DEPTH_SWEEP_INPUT_DIM}, "
              f"latent_dim={DEPTH_SWEEP_LATENT_DIM}, epochs={DEPTH_SWEEP_EPOCHS})")
        print("=" * 70)

        X_train, X_val, X_test = build_dataset_for_input_dim(DEPTH_SWEEP_INPUT_DIM, nonlinear=nonlinear)

        for n_layers in HIDDEN_LAYER_COUNTS:
            model, mse = train_one_config(
                X_train, X_val, X_test, DEPTH_SWEEP_INPUT_DIM, DEPTH_SWEEP_LATENT_DIM,
                n_layers, n_layers, epochs=DEPTH_SWEEP_EPOCHS,
            )
            metrics = evaluate_model(model, DEPTH_SWEEP_INPUT_DIM, mse)

            out.append({
                "sweep": tag, "input_dim": DEPTH_SWEEP_INPUT_DIM, "latent_dim": DEPTH_SWEEP_LATENT_DIM,
                "encoder_hidden_layers": n_layers, "decoder_hidden_layers": n_layers,
                "compression_ratio": DEPTH_SWEEP_INPUT_DIM / DEPTH_SWEEP_LATENT_DIM, **metrics,
            })
            print(f"  hidden_layers={n_layers}  MSE={mse:.6f}  params={metrics['n_params']:,}  "
                  f"MACs={metrics['macs']:,}  latency={metrics['latency_ms']:.4f} ms  "
                  f"energy~{metrics['energy_est_nJ']:.1f} nJ")
    return out


def run_sweep_c() -> List[dict]:
    """encoder/decoder hidden layers varied independently -- the asymmetric
    autoencoder question. Same (input_dim, latent_dim) and epoch budget as
    Sweep B so results are directly comparable to the symmetric depth sweep."""
    out: List[dict] = []

    for nonlinear in (False, True):
        tag = "asym_nonlinear" if nonlinear else "asym_linear"
        print("\n" + "=" * 70)
        print(f"[pythae] SWEEP C — {tag}  (input_dim={ASYM_INPUT_DIM}, "
              f"latent_dim={ASYM_LATENT_DIM}, epochs={DEPTH_SWEEP_EPOCHS})")
        print("=" * 70)

        X_train, X_val, X_test = build_dataset_for_input_dim(ASYM_INPUT_DIM, nonlinear=nonlinear)

        for encoder_layers, decoder_layers in ASYM_CONFIGS:
            model, mse = train_one_config(
                X_train, X_val, X_test, ASYM_INPUT_DIM, ASYM_LATENT_DIM,
                encoder_layers, decoder_layers, epochs=DEPTH_SWEEP_EPOCHS,
            )
            metrics = evaluate_model(model, ASYM_INPUT_DIM, mse)

            out.append({
                "sweep": tag, "input_dim": ASYM_INPUT_DIM, "latent_dim": ASYM_LATENT_DIM,
                "encoder_hidden_layers": encoder_layers, "decoder_hidden_layers": decoder_layers,
                "compression_ratio": ASYM_INPUT_DIM / ASYM_LATENT_DIM, **metrics,
            })
            print(f"  encoder_layers={encoder_layers}  decoder_layers={decoder_layers}  MSE={mse:.6f}  "
                  f"params={metrics['n_params']:,}  encoder_MACs={metrics['encoder_macs']:,}  "
                  f"decoder_MACs={metrics['decoder_macs']:,}  latency={metrics['latency_ms']:.4f} ms")
    return out


def save_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# Palette / chart-chrome tokens from the dataviz skill's reference palette.

COLOR_PAGE = "#f9f9f7"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def _style_axis(ax) -> None:
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.label.set_color(COLOR_TEXT_SECONDARY)


def _label_bars(ax, xs, ys, fmt: str) -> None:
    for xv, yv in zip(xs, ys):
        ax.annotate(fmt.format(yv), xy=(xv, yv), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_SECONDARY)


def make_dimension_depth_plot(rows: List[dict], path: Path) -> None:
    dim_rows = [r for r in rows if r["sweep"] == "dimension"]
    depth_lin_rows = sorted((r for r in rows if r["sweep"] == "depth_linear"), key=lambda r: r["encoder_hidden_layers"])
    depth_nl_rows = sorted((r for r in rows if r["sweep"] == "depth_nonlinear"), key=lambda r: r["encoder_hidden_layers"])

    fig = plt.figure(figsize=(13, 8.5), facecolor=COLOR_PAGE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.55, wspace=0.32)

    ax0 = fig.add_subplot(gs[:, 0])
    _style_axis(ax0)
    input_dims = sorted({r["input_dim"] for r in dim_rows})
    for i, input_dim in enumerate(input_dims):
        series = sorted((r for r in dim_rows if r["input_dim"] == input_dim), key=lambda r: r["compression_ratio"])
        x = [r["compression_ratio"] for r in series]
        y = [r["mse"] for r in series]
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax0.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                 markeredgewidth=1.5, markeredgecolor=COLOR_SURFACE, label=f"input_dim={input_dim}", zorder=3)
        ax0.annotate(str(input_dim), xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                     fontsize=9, va="center", zorder=4)
    ax0.set_xscale("log", base=2)
    ax0.set_yscale("log")
    ax0.set_xlabel("Compression ratio  (input_dim / latent_dim)")
    ax0.set_ylabel("Reconstruction MSE (test)")
    ax0.set_title("[pythae] Compression ratio vs. MSE, by input dimension", fontsize=11, loc="left")
    ax0.legend(frameon=False, fontsize=9, loc="upper left")

    layers = [r["encoder_hidden_layers"] for r in depth_lin_rows]

    ax1 = fig.add_subplot(gs[0, 1])
    _style_axis(ax1)
    macs_vals = [r["macs"] for r in depth_lin_rows]
    ax1.bar(layers, macs_vals, color=CATEGORICAL[0], width=0.6, zorder=3)
    _label_bars(ax1, layers, macs_vals, "{:,.0f}")
    ax1.margins(y=0.18)
    ax1.set_ylabel("MACs / forward pass")
    ax1.set_title(f"[pythae] Compute cost vs. depth  (in={DEPTH_SWEEP_INPUT_DIM}, latent={DEPTH_SWEEP_LATENT_DIM})",
                  fontsize=10, loc="left")
    ax1.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    _style_axis(ax2)
    lat_vals = [r["latency_ms"] for r in depth_lin_rows]
    ax2.bar(layers, lat_vals, color=CATEGORICAL[1], width=0.6, zorder=3)
    _label_bars(ax2, layers, lat_vals, "{:.3f}")
    ax2.margins(y=0.18)
    ax2.set_ylabel("Latency (ms)")
    ax2.tick_params(labelbottom=False)

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    _style_axis(ax3)
    offset = 0.18
    mse_lin = [r["mse"] for r in depth_lin_rows]
    mse_nl = [r["mse"] for r in depth_nl_rows]
    ax3.bar([x - offset for x in layers], mse_lin, width=2 * offset,
            color=CATEGORICAL[4], label="linear data", zorder=3)
    ax3.bar([x + offset for x in layers], mse_nl, width=2 * offset,
            color=CATEGORICAL[5], label="nonlinear data", zorder=3)
    _label_bars(ax3, [x - offset for x in layers], mse_lin, "{:.4f}")
    _label_bars(ax3, [x + offset for x in layers], mse_nl, "{:.4f}")
    ax3.margins(y=0.22)
    ax3.set_ylabel("MSE (test)")
    ax3.set_xlabel("Hidden layers (encoder = decoder depth)")
    ax3.set_xticks(layers)
    ax3.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

    fig.suptitle("pythae AE — architecture sweep (dimension + depth)", fontsize=13, y=0.98)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_asymmetric_plot(rows: List[dict], path: Path) -> None:
    lin_rows = {(r["encoder_hidden_layers"], r["decoder_hidden_layers"]): r
                for r in rows if r["sweep"] == "asym_linear"}
    nl_rows = {(r["encoder_hidden_layers"], r["decoder_hidden_layers"]): r
               for r in rows if r["sweep"] == "asym_nonlinear"}

    labels = [f"enc{e}/dec{d}" for e, d in ASYM_CONFIGS]
    x = list(range(len(ASYM_CONFIGS)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=COLOR_PAGE)

    _style_axis(ax0)
    offset = 0.18
    mse_lin = [lin_rows[cfg]["mse"] for cfg in ASYM_CONFIGS]
    mse_nl = [nl_rows[cfg]["mse"] for cfg in ASYM_CONFIGS]
    ax0.bar([i - offset for i in x], mse_lin, width=2 * offset, color=CATEGORICAL[4], label="linear data", zorder=3)
    ax0.bar([i + offset for i in x], mse_nl, width=2 * offset, color=CATEGORICAL[5], label="nonlinear data", zorder=3)
    _label_bars(ax0, [i - offset for i in x], mse_lin, "{:.4f}")
    _label_bars(ax0, [i + offset for i in x], mse_nl, "{:.4f}")
    ax0.margins(y=0.22)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("MSE (test)")
    ax0.set_title("[pythae] Asymmetric AE — MSE by architecture", fontsize=11, loc="left")
    ax0.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

    _style_axis(ax1)
    enc_macs = [lin_rows[cfg]["encoder_macs"] for cfg in ASYM_CONFIGS]
    dec_macs = [lin_rows[cfg]["decoder_macs"] for cfg in ASYM_CONFIGS]
    ax1.bar([i - offset for i in x], enc_macs, width=2 * offset, color=CATEGORICAL[0], label="encoder MACs", zorder=3)
    ax1.bar([i + offset for i in x], dec_macs, width=2 * offset, color=CATEGORICAL[2], label="decoder MACs", zorder=3)
    _label_bars(ax1, [i - offset for i in x], enc_macs, "{:,.0f}")
    _label_bars(ax1, [i + offset for i in x], dec_macs, "{:,.0f}")
    ax1.margins(y=0.22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("MACs / forward pass")
    ax1.set_title("[pythae] Where the compute cost sits", fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

    fig.suptitle(f"pythae AE — asymmetric encoder/decoder depth  "
                 f"(input_dim={ASYM_INPUT_DIM}, latent_dim={ASYM_LATENT_DIM})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = run_sweep_a() + run_sweep_b() + run_sweep_c()
    save_csv(rows, CSV_PATH)
    make_dimension_depth_plot(rows, PLOT_PATH)
    make_asymmetric_plot(rows, ASYM_PLOT_PATH)
    print(f"Saved plots to {PLOT_PATH} and {ASYM_PLOT_PATH}")


if __name__ == "__main__":
    main()
