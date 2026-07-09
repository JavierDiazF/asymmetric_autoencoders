"""
Example: symbol-wise regression AE for a simplified optical-fiber link.

Survey baseline (Alnaseri et al., 2026), Sec. V-B: dense encoder/decoder,
ELU hidden activations (papers such as [8] Uhlemann et al. switch from
ReLU to ELU specifically for fiber), linear regression output, MSE loss,
trained over a *differentiable* channel model.

IMPORTANT LIMITATION (flagged explicitly by the survey, Sec. V-B-3): a
feedforward dense AE over a simplified channel is a reasonable compromise
for short-block experiments, but it does not capture the long memory of
real fiber links (chromatic dispersion spreads energy across many symbols)
or Kerr nonlinearity. The channel used here (OpticalFiberChannel, see
channel.py) is a linear FIR + AWGN proxy for dispersion-induced ISI, NOT a
split-step Fourier (SSFM) nonlinear fiber model. For long channel memory,
the survey points to recurrent architectures (BRNN/SBRNN) instead -- a
natural next step once this dense baseline is validated.

Message representation: identical convention to example_wireless.py --
N_SYMBOLS complex constellation points as 2*N_SYMBOLS reals (I rail then
Q rail). A longer block is used here than in the wireless example so the
FIR-filter ISI has enough neighbouring symbols to spread into.

Pipeline (Tx/Rx split, same pattern as the rest of this repo):
  Tx      : symbols -> encoder -> power-norm -> w      (deployed at the transmitter)
  Channel : w -> ISI (FIR) -> AWGN(snr_db) -> y         (the fiber link)
  Rx      : y -> decoder -> reconstructed symbols        (deployed at the receiver)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AsymmetricAutoencoder
from channel import OpticalFiberChannel, PowerNormalization
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train


# ── Config — change these to experiment with dim / depth ─────────────────────

N_SYMBOLS   = 8                  # channel uses (complex symbols) per block
INPUT_DIM   = 2 * N_SYMBOLS      # I/Q concatenated -> change freely
LATENT_DIM  = 2 * N_SYMBOLS      # < INPUT_DIM compresses, > INPUT_DIM adds redundancy
HIDDEN_DIMS_ENCODER = [64, 64]   # number of hidden layers = len(...)
HIDDEN_DIMS_DECODER = [64, 64]
ACTIVATION  = "elu"              # survey Sec. V-B: [8] found ELU better than ReLU for fiber

CONSTELLATION      = "qpsk"      # "qpsk" | "16qam"
ISI_TAPS           = [0.05, 0.15, 0.9, 0.15, 0.05]   # symmetric FIR proxy for CD-induced ISI
TRAIN_SNR_DB_RANGE  = (5.0, 25.0)   # domain randomization, survey Sec. IV-C
EVAL_SNR_DB_LIST   = [5, 10, 15, 20, 25]

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic constellation data ──────────────────────────────────────────────

def qpsk_points() -> np.ndarray:
    a = 1 / np.sqrt(2)
    return np.array([[a, a], [a, -a], [-a, a], [-a, -a]], dtype=np.float32)


def qam16_points() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=np.float32) / np.sqrt(10)
    return np.array([[i, q] for i in levels for q in levels], dtype=np.float32)


CONSTELLATIONS = {"qpsk": qpsk_points, "16qam": qam16_points}


def sample_symbols(n_samples: int, n_symbols: int, constellation: str) -> np.ndarray:
    """Returns (n_samples, 2*n_symbols): I rail (n_symbols) then Q rail (n_symbols)."""
    points = CONSTELLATIONS[constellation]()
    idx = np.random.randint(0, len(points), size=(n_samples, n_symbols))
    iq = points[idx]  # (n_samples, n_symbols, 2)
    return np.concatenate([iq[..., 0], iq[..., 1]], axis=-1).astype(np.float32)


N_TRAIN, N_VAL, N_TEST = 20_000, 2_000, 5_000

X_train = torch.tensor(sample_symbols(N_TRAIN, N_SYMBOLS, CONSTELLATION))
X_val   = torch.tensor(sample_symbols(N_VAL,   N_SYMBOLS, CONSTELLATION))
X_test  = torch.tensor(sample_symbols(N_TEST,  N_SYMBOLS, CONSTELLATION))

train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val),   batch_size=512)

print(f"Fiber AE  |  {CONSTELLATION.upper()}  |  {N_SYMBOLS} symbols/block "
      f"({INPUT_DIM} real dims)  |  latent {LATENT_DIM}  |  "
      f"hidden enc={HIDDEN_DIMS_ENCODER} dec={HIDDEN_DIMS_DECODER}  |  ISI taps={ISI_TAPS}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING  (encoder -> power-norm -> ISI+AWGN(random SNR) -> decoder)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 1 — Training")
print("=" * 65)

encoder = Encoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS_ENCODER,
    activation=ACTIVATION,
    latent_activation="none",   # linear encoder output, Fig. 2a
)
decoder = Decoder(
    latent_dim=LATENT_DIM,
    output_dim=INPUT_DIM,
    hidden_dims=HIDDEN_DIMS_DECODER,
    activation=ACTIVATION,
    output_activation="none",   # regression output, Fig. 2a (unbounded, not [0,1])
)
train_channel = nn.Sequential(
    PowerNormalization(),
    OpticalFiberChannel(n_symbols=N_SYMBOLS, snr_db_range=TRAIN_SNR_DB_RANGE, isi_taps=ISI_TAPS),
)

model = AsymmetricAutoencoder(encoder, decoder, channel=train_channel).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-3)

cfg = TrainConfig(
    epochs=150,
    loss="mse",
    early_stopping_patience=15,
    early_stopping_min_delta=1e-6,
    log_every=10,
    device=str(DEVICE),
)

train(model, train_loader, optimizer, cfg, val_loader)

torch.save(model.encoder.state_dict(), WEIGHTS_DIR / "encoder_fiber.pt")
torch.save(model.decoder.state_dict(), WEIGHTS_DIR / "decoder_fiber.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRANSMITTER  (encoder + power-norm only)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 2 — Transmitter: encoding test symbols")
print("=" * 65)

tx_encoder = Encoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS_ENCODER,
    activation=ACTIVATION,
    latent_activation="none",
).to(DEVICE)
tx_encoder.load_state_dict(torch.load(WEIGHTS_DIR / "encoder_fiber.pt", weights_only=True))
tx_encoder.eval()

tx_power_norm = PowerNormalization()

with torch.no_grad():
    w = tx_power_norm(tx_encoder(X_test.to(DEVICE)))

print(f"  Transmitted signal: {tuple(X_test.shape)} -> {tuple(w.shape)}  "
      f"(avg energy/sample: {w.pow(2).sum(-1).mean().item():.3f}, target={LATENT_DIM})")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 & 4 — FIBER CHANNEL (ISI + AWGN) + RECEIVER, swept over SNR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 3/4 — Channel + Receiver: SNR sweep")
print("=" * 65)

rx_decoder = Decoder(
    latent_dim=LATENT_DIM,
    output_dim=INPUT_DIM,
    hidden_dims=HIDDEN_DIMS_DECODER,
    activation=ACTIVATION,
    output_activation="none",
).to(DEVICE)
rx_decoder.load_state_dict(torch.load(WEIGHTS_DIR / "decoder_fiber.pt", weights_only=True))
rx_decoder.eval()

points = torch.tensor(CONSTELLATIONS[CONSTELLATION](), device=DEVICE)
x_true = X_test.to(DEVICE)
true_iq = torch.stack([x_true[:, :N_SYMBOLS], x_true[:, N_SYMBOLS:]], dim=-1)  # (N, n_symbols, 2)

print(f"\n{'SNR (dB)':>10}  {'MSE':>12}  {'Symbol err. rate*':>18}")

results = {}
for snr_db in EVAL_SNR_DB_LIST:
    eval_channel = OpticalFiberChannel(n_symbols=N_SYMBOLS, snr_db=snr_db, isi_taps=ISI_TAPS).to(DEVICE)
    with torch.no_grad():
        y = eval_channel(w)
        x_hat = rx_decoder(y)

        mse = (x_hat - x_true).pow(2).mean().item()

        hat_iq = torch.stack([x_hat[:, :N_SYMBOLS], x_hat[:, N_SYMBOLS:]], dim=-1)
        dists = torch.cdist(hat_iq.reshape(-1, 2), points)
        nearest = points[dists.argmin(dim=-1)].reshape(hat_iq.shape)
        ser = ((nearest - true_iq).abs().sum(dim=-1) > 1e-4).float().mean().item()

    results[snr_db] = (mse, ser)
    print(f"{snr_db:>10}  {mse:>12.6f}  {ser:>18.4%}")

print("\n* nearest-constellation-point mismatch rate: an approximate proxy for")
print("  symbol error rate, not a full ML/MAP detector.")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"SUMMARY  —  {CONSTELLATION.upper()}  {INPUT_DIM} -> {LATENT_DIM}  "
      f"(hidden enc={HIDDEN_DIMS_ENCODER} dec={HIDDEN_DIMS_DECODER}, ISI taps={ISI_TAPS})")
print("=" * 65)
print(f"{'SNR (dB)':<10}  {'MSE':>12}  {'SER (approx.)':>15}")
for snr_db, (mse, ser) in results.items():
    print(f"{snr_db:<10}  {mse:>12.6f}  {ser:>15.4%}")

print("\nNOTE: OpticalFiberChannel is a linear ISI+AWGN proxy, not a full SSFM")
print("      nonlinear fiber model -- see module docstring / survey Sec. V-B.")
