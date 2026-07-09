"""
Architecture sweep for the *compression-only* autoencoder (no channel).

Context: example_wireless.py and example_fiber.py differ only in the
channel.py piece placed between encoder and decoder (AWGN vs. ISI+AWGN).
Drop the channel and the two collapse into the same model: dense
encoder -> decoder, MSE loss -- exactly what this script sweeps. It is not
tied to wireless/fiber/IoT specifically; it studies how the shared
Encoder/Decoder building blocks (encoder.py, decoder.py, autoencoder.py,
train.py) trade off compression ratio, reconstruction MSE, and an
energy proxy as input_dim, latent_dim, and hidden-layer count change.

Two sweeps:
  A) DIMENSION  -- for several input_dim values, sweep the compression
     ratio (input_dim / latent_dim) at a fixed depth. Answers: "what does
     changing input_dim and latent_dim do to achievable MSE at a given
     compression ratio?"
  B) DEPTH       -- at a fixed (input_dim, latent_dim), sweep the number of
     hidden layers (encoder and decoder built symmetrically via
     interpolate_widths), run once on linear data and once on nonlinear data
     (see build_dataset_for_input_dim). Answers: "what does adding hidden
     layers cost in compute/energy, and when does the extra MSE reduction
     actually pay for that cost?" Uses a much larger epoch budget than
     Sweep A (DEPTH_SWEEP_EPOCHS) -- deeper nets need more optimisation
     steps to converge, so reusing Sweep A's budget would just be measuring
     which model finished training in time, not which model fits best.

Synthetic data: i.i.d. noise has no structure to compress, so instead each
sample is drawn from a controlled low-rank generative model
    X = Z @ A + noise,   Z ~ N(0, I_TRUE_RANK),  A: TRUE_RANK x input_dim
      (columns of A unit-norm, so every feature has ~unit signal variance
      regardless of input_dim -- keeps per-feature SNR comparable across
      the whole input_dim sweep)
so TRUE_RANK is the data's intrinsic dimensionality: an AE with
latent_dim >= TRUE_RANK can in principle reconstruct near-perfectly, while
latent_dim < TRUE_RANK necessarily loses information. This gives an
interpretable "elbow" in the MSE-vs-compression-ratio curves.

Energy proxy: there is no wattmeter here, so "energy" is estimated the way
Sec. VI of the survey quantifies AE complexity -- MACs (multiply-accumulate
ops), i.e. sum(in_features * out_features) over every nn.Linear layer -- plus
measured wall-clock inference latency, and a rough compute-only energy
estimate using ~4.6 pJ/MAC (Horowitz, ISSCC 2014, 45nm float32 mult+add).
That figure excludes memory-access energy, which typically dominates on
real edge hardware, so n_params (proxy for memory traffic) is reported
alongside it -- treat these as order-of-magnitude proxies, not measurements.

Outputs (in results/):
  compression_sweep.csv  -- one row per trained configuration
  compression_sweep.png  -- combined figure for both sweeps
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AsymmetricAutoencoder
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train


# ── Config — change these to experiment ───────────────────────────────────────

INPUT_DIMS               = [32, 64, 128, 256, 512, 1024]     # Sweep A
COMPRESSION_RATIOS       = [2, 4, 8, 16, 32]        # latent_dim = input_dim // ratio
HIDDEN_LAYERS_DIM_SWEEP  = 2                        # depth held fixed during Sweep A

DEPTH_SWEEP_INPUT_DIM    = 256                      # Sweep B (fixed dims)
DEPTH_SWEEP_LATENT_DIM   = 32                       # ratio 8, safely above TRUE_RANK
HIDDEN_LAYER_COUNTS      = [0, 1, 2, 3, 4]
# Deeper nets need more optimisation steps to converge -- reusing Sweep A's
# epoch budget systematically penalises them for not having finished
# training, not for lacking capacity. Give Sweep B a much larger budget so
# the depth comparison is actually fair.
DEPTH_SWEEP_EPOCHS       = 300
DEPTH_SWEEP_PATIENCE     = 30

TRUE_RANK  = 12       # intrinsic dimensionality of the synthetic data
NOISE_STD  = 0.05      # additive observation noise (absolute, on ~unit-variance features)
NONLINEAR_HIDDEN = 24  # width of the fixed random generator used for the nonlinear variant

ACTIVATION = "elu"
EPOCHS     = 60
N_TRAIN, N_VAL, N_TEST = 8_000, 1_000, 2_000

ENERGY_PJ_PER_MAC = 4.6   # Horowitz ISSCC'14, 45nm float32 mult+add -- compute-only estimate

RESULTS_DIR = Path("results")
CSV_PATH  = RESULTS_DIR / "compression_sweep.csv"
PLOT_PATH = RESULTS_DIR / "compression_sweep.png"

FIELDNAMES = [
    "sweep", "input_dim", "latent_dim", "hidden_layers", "compression_ratio",
    "mse", "n_params", "macs", "energy_est_nJ", "latency_ms",
]

torch.manual_seed(42)
rng = np.random.default_rng(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic low-rank data ────────────────────────────────────────────────────

def build_dataset_for_input_dim(
    input_dim: int, nonlinear: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank synthetic data, same generator shared across train/val/test so
    the structure is actually learnable (a fresh generator per split would
    make the splits statistically unrelated).

    nonlinear=False (default, used by Sweep A and half of Sweep B):
        X = Z @ A + noise            -- a LINEAR map. A single Linear layer
        (0 hidden layers) is already the matched model class for this, so
        adding hidden layers cannot reduce MSE, only cost more to train.

    nonlinear=True (the other half of Sweep B):
        X = tanh(Z @ W1) @ W2 + noise -- Z is pushed through a small fixed
        random 1-hidden-layer network before the final linear projection, so
        the encoder->decoder map a linear (0-layer) AE would need to invert
        is genuinely curved. This is the case where extra depth should
        actually pay for itself in lower MSE.
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


# ── Architecture / cost helpers ────────────────────────────────────────────────

def interpolate_widths(start: int, end: int, n: int) -> List[int]:
    """n geometrically-spaced interior widths between start and end (both
    exclusive). n=0 -> [] (direct start->end projection, the AAE-0 style
    already used in example.py)."""
    if n <= 0:
        return []
    sizes = np.geomspace(start, end, num=n + 2)[1:-1]
    return [max(1, int(round(s))) for s in sizes]


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_macs(module: nn.Module) -> int:
    """MAC count = sum(in_features * out_features) over every nn.Linear layer
    -- the same formula the survey uses in Sec. VI-A-1 (Eq. 9-10) for dense
    encoder/decoder inference complexity."""
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


def measure_latency_ms(model: nn.Module, input_dim: int, device: torch.device,
                        batch_size: int = 1, n_warmup: int = 20, n_repeats: int = 200) -> float:
    """Empirical wall-clock forward-pass latency (encode+decode), averaged over
    n_repeats after a warmup period."""
    model.eval()
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_repeats * 1000.0


def train_one_config(
    X_train: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor,
    input_dim: int, latent_dim: int, n_layers: int,
    epochs: int = EPOCHS, patience: int = 8,
) -> Tuple[AsymmetricAutoencoder, float]:
    """Builds, trains (no channel -- pure compression) and evaluates one
    (input_dim, latent_dim, n_layers) configuration."""
    encoder = Encoder(
        input_dim=input_dim, latent_dim=latent_dim,
        hidden_dims=interpolate_widths(input_dim, latent_dim, n_layers),
        activation=ACTIVATION, latent_activation="selu",
    )
    decoder = Decoder(
        latent_dim=latent_dim, output_dim=input_dim,
        hidden_dims=interpolate_widths(latent_dim, input_dim, n_layers),
        activation=ACTIVATION, output_activation="sigmoid",
    )
    model = AsymmetricAutoencoder(encoder, decoder).to(DEVICE)  # channel=None: pure compression

    optimizer = Adam(model.parameters(), lr=2e-3)
    cfg = TrainConfig(
        epochs=epochs, loss="mse", early_stopping_patience=patience,
        early_stopping_min_delta=1e-7, log_every=0, device=str(DEVICE),
    )

    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=256)

    train(model, train_loader, optimizer, cfg, val_loader)

    model.eval()
    with torch.no_grad():
        x_hat, _ = model(X_test.to(DEVICE))
        test_mse = (x_hat - X_test.to(DEVICE)).pow(2).mean().item()

    return model, test_mse


def evaluate_model(model: AsymmetricAutoencoder, input_dim: int, mse: float) -> Dict[str, float]:
    n_params = count_params(model)
    macs = count_macs(model)
    return {
        "mse": mse,
        "n_params": n_params,
        "macs": macs,
        "energy_est_nJ": macs * ENERGY_PJ_PER_MAC / 1000.0,
        "latency_ms": measure_latency_ms(model, input_dim, DEVICE),
    }


def run_sweep_a() -> List[dict]:
    """input_dim x compression ratio, fixed depth."""
    print("=" * 70)
    print(f"SWEEP A — dimension  (hidden_layers fixed at {HIDDEN_LAYERS_DIM_SWEEP})")
    print("=" * 70)

    out: List[dict] = []
    for input_dim in INPUT_DIMS:
        X_train, X_val, X_test = build_dataset_for_input_dim(input_dim)

        for ratio in COMPRESSION_RATIOS:
            latent_dim = max(1, input_dim // ratio)
            model, mse = train_one_config(X_train, X_val, X_test, input_dim, latent_dim, HIDDEN_LAYERS_DIM_SWEEP)
            metrics = evaluate_model(model, input_dim, mse)

            out.append({
                "sweep": "dimension", "input_dim": input_dim, "latent_dim": latent_dim,
                "hidden_layers": HIDDEN_LAYERS_DIM_SWEEP,
                "compression_ratio": input_dim / latent_dim, **metrics,
            })
            print(f"  input_dim={input_dim:>4}  ratio={ratio:>3}  latent_dim={latent_dim:>4}  "
                  f"MSE={mse:.6f}  params={metrics['n_params']:,}  MACs={metrics['macs']:,}")
    return out


def run_sweep_b() -> List[dict]:
    """hidden-layer count, fixed input_dim / latent_dim, run once on linear
    data (where a 0-layer AE is already the matched model) and once on
    nonlinear data (where depth should genuinely help) -- see
    build_dataset_for_input_dim's docstring. Both use DEPTH_SWEEP_EPOCHS,
    much larger than Sweep A's budget, so deeper nets get a fair chance to
    converge instead of being penalised for simply not finishing training."""
    out: List[dict] = []

    for nonlinear in (False, True):
        tag = "depth_nonlinear" if nonlinear else "depth_linear"
        print("\n" + "=" * 70)
        print(f"SWEEP B — {tag}  (input_dim={DEPTH_SWEEP_INPUT_DIM}, latent_dim={DEPTH_SWEEP_LATENT_DIM}, "
              f"epochs<={DEPTH_SWEEP_EPOCHS})")
        print("=" * 70)

        X_train, X_val, X_test = build_dataset_for_input_dim(DEPTH_SWEEP_INPUT_DIM, nonlinear=nonlinear)

        for n_layers in HIDDEN_LAYER_COUNTS:
            model, mse = train_one_config(
                X_train, X_val, X_test, DEPTH_SWEEP_INPUT_DIM, DEPTH_SWEEP_LATENT_DIM, n_layers,
                epochs=DEPTH_SWEEP_EPOCHS, patience=DEPTH_SWEEP_PATIENCE,
            )
            metrics = evaluate_model(model, DEPTH_SWEEP_INPUT_DIM, mse)

            out.append({
                "sweep": tag, "input_dim": DEPTH_SWEEP_INPUT_DIM, "latent_dim": DEPTH_SWEEP_LATENT_DIM,
                "hidden_layers": n_layers,
                "compression_ratio": DEPTH_SWEEP_INPUT_DIM / DEPTH_SWEEP_LATENT_DIM, **metrics,
            })
            print(f"  hidden_layers={n_layers}  MSE={mse:.6f}  params={metrics['n_params']:,}  "
                  f"MACs={metrics['macs']:,}  latency={metrics['latency_ms']:.4f} ms  "
                  f"energy~{metrics['energy_est_nJ']:.1f} nJ")
    return out


def save_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {path}")


def load_csv(path: Path) -> List[dict]:
    """Reloads a previously saved sweep, e.g. to redraw the plot without
    retraining anything."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for r in reader:
            row = {k: (int(v) if k in ("input_dim", "latent_dim", "hidden_layers", "n_params", "macs") else
                       float(v) if k not in ("sweep",) else v)
                   for k, v in r.items()}
            out.append(row)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════
# Palette / chart-chrome tokens from the dataviz skill's reference palette
# (references/palette.md): fixed-order categorical hues, hairline solid grid,
# no dual-axis charts -- energy/latency/MSE vs. depth are three separate
# stacked panels rather than one panel with multiple y-scales.

COLOR_PAGE = "#f9f9f7"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_PRIMARY = "#0b0b0b"
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
    ax.title.set_color(COLOR_TEXT_PRIMARY)


def _label_bars(ax, xs, ys, fmt: str) -> None:
    for xv, yv in zip(xs, ys):
        ax.annotate(fmt.format(yv), xy=(xv, yv), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_SECONDARY)


def make_plot(rows: List[dict], path: Path) -> None:
    dim_rows = [r for r in rows if r["sweep"] == "dimension"]
    depth_lin_rows = sorted((r for r in rows if r["sweep"] == "depth_linear"), key=lambda r: r["hidden_layers"])
    depth_nl_rows = sorted((r for r in rows if r["sweep"] == "depth_nonlinear"), key=lambda r: r["hidden_layers"])

    fig = plt.figure(figsize=(13, 8.5), facecolor=COLOR_PAGE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.55, wspace=0.32)

    # ── Left: MSE vs compression ratio, one line per input_dim ──────────
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
                     color=COLOR_TEXT_SECONDARY, fontsize=9, va="center", zorder=4)
    ax0.set_xscale("log", base=2)
    ax0.set_yscale("log")
    ax0.set_xlabel("Compression ratio  (input_dim / latent_dim)")
    ax0.set_ylabel("Reconstruction MSE (test)")
    ax0.set_title("Compression ratio vs. MSE, by input dimension", fontsize=11, loc="left")
    # upper-left is the one corner no series passes through (all start near
    # the bottom at ratio=2) -- avoids the legend box colliding with the
    # direct end-labels clustered at the bottom-right of the log-log plot.
    ax0.legend(frameon=False, fontsize=9, loc="upper left")

    # ── Right column: energy proxies + MSE vs hidden-layer count ────────
    # MACs/latency depend only on the architecture (n_layers), not on which
    # data variant trained it, so the linear-sweep rows are a fine source
    # for both -- values are identical to the nonlinear-sweep rows.
    layers = [r["hidden_layers"] for r in depth_lin_rows]

    ax1 = fig.add_subplot(gs[0, 1])
    _style_axis(ax1)
    macs_vals = [r["macs"] for r in depth_lin_rows]
    ax1.bar(layers, macs_vals, color=CATEGORICAL[0], width=0.6, zorder=3)
    _label_bars(ax1, layers, macs_vals, "{:,.0f}")
    ax1.margins(y=0.18)
    ax1.set_ylabel("MACs / forward pass")
    ax1.set_title(f"Compute cost vs. depth  (in={DEPTH_SWEEP_INPUT_DIM}, latent={DEPTH_SWEEP_LATENT_DIM})",
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

    # MSE: linear vs nonlinear data, grouped bars -- this is the "does depth
    # actually pay for itself" comparison (see build_dataset_for_input_dim).
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

    fig.suptitle("Compression autoencoder — architecture sweep", fontsize=13, color=COLOR_TEXT_PRIMARY, y=0.98)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = run_sweep_a() + run_sweep_b()
    save_csv(rows, CSV_PATH)
    make_plot(rows, PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
