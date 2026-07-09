"""
Comparison: AAE-0 to AAE-4 (Gilbert et al., 2024) on real smart-home data.

Dataset: HomeC.csv — 503K rows of smart-home sensor readings.

Each sample fed to the autoencoder is a SLIDING WINDOW of WINDOW_SIZE
consecutive rows, flattened to a 1D vector:
    INPUT_DIM = WINDOW_SIZE x n_features  (e.g. 100 x 28 = 2800)
    LATENT_DIM = INPUT_DIM // 4           (compression ratio 4:1)

Pipeline:
  Sensor : window of rows -> normalize -> flatten -> encode -> transmit z
  Cloud  : receive z -> decode -> unflatten -> desnormalize -> original units
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AsymmetricAutoencoder
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train


# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH    = Path("smart-home-dataset/HomeC.csv")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

SKIP_COLS = {"time", "icon", "summary", "cloudCover"}

WINDOW_SIZE     = 100   # number of consecutive rows per sample
STRIDE          = 50    # step between windows (50 = 50% overlap)
BYTES_PER_FLOAT = 4

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load CSV ──────────────────────────────────────────────────────────────────

print("Loading data from CSV...")

with open(CSV_PATH, newline="") as f:
    reader    = csv.reader(f)
    header    = next(reader)
    col_idx   = [i for i, h in enumerate(header) if h not in SKIP_COLS]
    col_names = [header[i] for i in col_idx]

    rows = []
    for row in reader:
        try:
            rows.append([float(row[i]) for i in col_idx])
        except (ValueError, IndexError):
            continue

data = np.array(rows, dtype=np.float32)
N_ROWS, N_FEATURES = data.shape

INPUT_DIM  = WINDOW_SIZE * N_FEATURES
LATENT_DIM = INPUT_DIM // 4

print(f"  Rows loaded  : {N_ROWS:,}")
print(f"  Features/row : {N_FEATURES}  →  {col_names[:3]} ...")
print(f"  Window size  : {WINDOW_SIZE} rows × {N_FEATURES} features = {INPUT_DIM} values/sample")
print(f"  LATENT_DIM   : {LATENT_DIM}  (ratio {INPUT_DIM // LATENT_DIM}:1)")


# ── Normalise ─────────────────────────────────────────────────────────────────
# Fit scaler on raw rows (before windowing) so we can desnormalize per-feature.

scaler  = MinMaxScaler()
data_01 = scaler.fit_transform(data)   # shape (N_ROWS, N_FEATURES), values in [0,1]


# ── Sliding window ────────────────────────────────────────────────────────────

def make_windows(matrix: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Returns array of shape (n_windows, window_size * n_features)."""
    n = len(matrix)
    starts  = range(0, n - window_size + 1, stride)
    windows = [matrix[s : s + window_size].flatten() for s in starts]
    return np.array(windows, dtype=np.float32)


windows = make_windows(data_01, WINDOW_SIZE, STRIDE)
print(f"\n  Windows created: {len(windows):,}  (stride={STRIDE})")


# ── Train / val / test split ──────────────────────────────────────────────────

n       = len(windows)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

W_train = windows[:n_train]
W_val   = windows[n_train : n_train + n_val]
W_test  = windows[n_train + n_val:]

print(f"  Split → train: {len(W_train):,}  |  val: {len(W_val):,}  |  test: {len(W_test):,}\n")

X_train = torch.tensor(W_train)
X_val   = torch.tensor(W_val)
X_test  = torch.tensor(W_test)

train_loader = DataLoader(TensorDataset(X_train), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val),   batch_size=512)


# ── AAE decoder configs (proportional to Table 2 of the paper) ───────────────
# Paper: latent -> [2x, 3x, 4x] -> output. Same proportions applied here.

half   = INPUT_DIM // 2
three4 = INPUT_DIM * 3 // 4

AAE_CONFIGS = {
    "AAE-0": [],
    "AAE-1": [half],
    "AAE-2": [half, three4],
    "AAE-3": [INPUT_DIM * 2 // 5,  INPUT_DIM * 3 // 5,  INPUT_DIM * 4 // 5],
    "AAE-4": [INPUT_DIM * 3 // 10, INPUT_DIM * 2 // 5,  INPUT_DIM // 2,
              INPUT_DIM * 7 // 10],
}

TRAIN_CFG = TrainConfig(
    epochs=100,
    loss="mse",
    early_stopping_patience=10,
    early_stopping_min_delta=1e-6,
    log_every=10,
    device=str(DEVICE),
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PHASE 1 — Training")
print("=" * 65)

for name, decoder_hidden in AAE_CONFIGS.items():
    arch = [LATENT_DIM] + decoder_hidden + [INPUT_DIM]
    print(f"\n  {name}  decoder: {' -> '.join(map(str, arch))}")

    encoder = Encoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=[],
        latent_activation="selu",
    )
    decoder = Decoder(
        latent_dim=LATENT_DIM,
        output_dim=INPUT_DIM,
        hidden_dims=decoder_hidden,
        activation="selu",
        output_activation="sigmoid",
    )

    model     = AsymmetricAutoencoder(encoder, decoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=3e-3)

    train(model, train_loader, optimizer, TRAIN_CFG, val_loader)

    torch.save(model.encoder.state_dict(), WEIGHTS_DIR / f"encoder_{name}.pt")
    torch.save(model.decoder.state_dict(), WEIGHTS_DIR / f"decoder_{name}.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SENSOR NODE  (encoder only)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 2 — Sensor node: encoding")
print("=" * 65)

bytes_original    = INPUT_DIM  * BYTES_PER_FLOAT
bytes_transmitted = LATENT_DIM * BYTES_PER_FLOAT
print(f"\n  Original (window)  : {bytes_original:,} bytes  ({WINDOW_SIZE} rows × {N_FEATURES} features × {BYTES_PER_FLOAT} B)")
print(f"  Compressed (z)     : {bytes_transmitted:,} bytes  ({LATENT_DIM} floats × {BYTES_PER_FLOAT} B)")
print(f"  Compression ratio  : {INPUT_DIM // LATENT_DIM}:1\n")

latent_codes = {}

for name in AAE_CONFIGS:
    sensor_encoder = Encoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=[],
        latent_activation="selu",
    )
    sensor_encoder.load_state_dict(
        torch.load(WEIGHTS_DIR / f"encoder_{name}.pt", weights_only=True)
    )
    sensor_encoder.eval()

    with torch.no_grad():
        z = sensor_encoder(X_test.to(DEVICE)).cpu().numpy()

    latent_codes[name] = z
    print(f"  {name}: {tuple(X_test.shape)} -> {z.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CLOUD SERVER  (decoder only)
# Unflatten window + desnormalize to compare in original units.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 3 — Cloud server: decoding and comparison")
print("=" * 65)

results = {}

for name, decoder_hidden in AAE_CONFIGS.items():
    cloud_decoder = Decoder(
        latent_dim=LATENT_DIM,
        output_dim=INPUT_DIM,
        hidden_dims=decoder_hidden,
        activation="selu",
        output_activation="sigmoid",
    )
    cloud_decoder.load_state_dict(
        torch.load(WEIGHTS_DIR / f"decoder_{name}.pt", weights_only=True)
    )
    cloud_decoder.eval()

    z = torch.tensor(latent_codes[name]).to(DEVICE)
    with torch.no_grad():
        x_hat_flat_01 = cloud_decoder(z).cpu().numpy()  # (n_windows, INPUT_DIM), in [0,1]

    # ── Unflatten: (n_windows, INPUT_DIM) -> (n_windows, WINDOW_SIZE, N_FEATURES)
    x_hat_unflat_01 = x_hat_flat_01.reshape(-1, WINDOW_SIZE, N_FEATURES)
    x_orig_unflat   = W_test.reshape(-1, WINDOW_SIZE, N_FEATURES)

    # ── Desnormalize each window row by row back to real units
    x_hat_real = np.stack([
        scaler.inverse_transform(w) for w in x_hat_unflat_01
    ])
    x_orig_real = np.stack([
        scaler.inverse_transform(w) for w in x_orig_unflat
    ])

    # ── Metrics in normalised scale and real units
    mse_norm = float(np.mean((x_hat_flat_01 - W_test) ** 2))
    mae_norm = float(np.mean(np.abs(x_hat_flat_01 - W_test)))
    mse_real = float(np.mean((x_hat_real - x_orig_real) ** 2))
    mae_real = float(np.mean(np.abs(x_hat_real - x_orig_real)))

    results[name] = (mse_norm, mae_norm, mse_real, mae_real)

    print(f"\n  {name}")
    print(f"    Normalised [0,1]  →  MSE: {mse_norm:.6f}  |  MAE: {mae_norm:.6f}")
    print(f"    Real units        →  MSE: {mse_real:.4f}   |  MAE: {mae_real:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"SUMMARY  —  window {WINDOW_SIZE}×{N_FEATURES}={INPUT_DIM} -> {LATENT_DIM}  (ratio {INPUT_DIM//LATENT_DIM}:1)")
print("=" * 65)
print(f"{'Model':<10}  {'MSE (norm)':>12}  {'MAE (norm)':>12}  {'MSE (real)':>12}  {'MAE (real)':>12}")
for name, (mse_n, mae_n, mse_r, mae_r) in results.items():
    print(f"{name:<10}  {mse_n:>12.6f}  {mae_n:>12.6f}  {mse_r:>12.4f}  {mae_r:>12.4f}")
