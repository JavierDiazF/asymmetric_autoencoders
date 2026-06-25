"""
Comparison: AAE-0 to AAE-4 (Gilbert et al., 2024) vs PCA.

All methods compress from INPUT_DIM=100 to LATENT_DIM=25 (ratio 4:1).
For each method we simulate the full sensor->cloud pipeline:
  - Sensor  : encodes x -> z  (only the encoder/transform is needed here)
  - Cloud   : decodes z -> x' (only the decoder/inverse-transform is needed here)

PCA is included as a classical linear baseline.
All results are compared in a final summary table.
"""
"""
NOTA:

Utiliza el selu para mejorar/superar las limitaciones que tiene relu como la polarización por omisión (0) de los valores menores que 0
SIn embargo tengo que mirar aún el tema de LeCun Normal Initialization que he visto que es con lo que mejor trabaja el selu.

Con respecto al sigmoid, era la función de activación por defetcto de los AEs. 
Sin embargo mirando funciones he visto que esta encaja más en clasificaciones binarias [0-1] y no en clasificaciones multiclase no sé si debería cambiarlo
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AsymmetricAutoencoder
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIM       = 100
LATENT_DIM      = 25
BYTES_PER_FLOAT = 4

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── AAE decoder architectures (Table 2 of the paper) ─────────────────────────

AAE_CONFIGS = {
    "AAE-0": [],
    "AAE-1": [50],
    "AAE-2": [50, 75],
    "AAE-3": [45, 65, 85],
    "AAE-4": [40, 55, 70, 85],
}

# ── Synthetic data (replace with your real sensor data) ──────────────────────
# Data is normalized to [0, 1] so the sigmoid output activation of the AAE
# and the PCA operate on the same scale.

N_TRAIN, N_VAL, N_TEST = 3000, 500, 500

raw_train = np.random.randn(N_TRAIN, INPUT_DIM).astype(np.float32)
raw_val   = np.random.randn(N_VAL,   INPUT_DIM).astype(np.float32)
raw_test  = np.random.randn(N_TEST,  INPUT_DIM).astype(np.float32)

scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(raw_train)   # fit only on train, apply to all
X_val_np   = scaler.transform(raw_val)
X_test_np  = scaler.transform(raw_test)

X_train = torch.tensor(X_train_np)
X_val   = torch.tensor(X_val_np)
X_test  = torch.tensor(X_test_np)

train_loader = DataLoader(TensorDataset(X_train), batch_size=50, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val),   batch_size=256)

TRAIN_CFG = TrainConfig(
    epochs=200,
    loss="mse",
    early_stopping_patience=15,
    early_stopping_min_delta=1e-5,
    log_every=0,
    device=str(DEVICE),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_mse_mae(original: np.ndarray, reconstructed: np.ndarray):
    diff = original - reconstructed
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    return mse, mae


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING  (offline, once, on a powerful machine)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PHASE 1 — Training all models")
print("=" * 65)

# ── Train AAE variants ────────────────────────────────────────────────────────

for name, decoder_hidden in AAE_CONFIGS.items():
    print(f"\n  Training {name}  (decoder hidden: {decoder_hidden or 'none'}) ...")

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

    # Save encoder and decoder weights independently
    torch.save(model.encoder.state_dict(), WEIGHTS_DIR / f"encoder_{name}.pt")
    torch.save(model.decoder.state_dict(), WEIGHTS_DIR / f"decoder_{name}.pt")
    print(f"  Saved encoder and decoder weights for {name}")

# ── Fit PCA ───────────────────────────────────────────────────────────────────
# PCA encoder  = projection onto the top LATENT_DIM principal components
# PCA decoder  = inverse projection back to original space
# Both live in the same PCA object, but in a real deployment you could ship
# only pca.components_ and pca.mean_ to the sensor (for the encode step).

print(f"\n  Fitting PCA  (n_components={LATENT_DIM}) ...")
pca = PCA(n_components=LATENT_DIM) # Reduce input dimension to LATENT dimension used in autoencoders
pca.fit(X_train_np) # En esta línea dudo de si está bien hecho

with open(WEIGHTS_DIR / "pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SENSOR NODE  (encoder / transform only)
# Compresses the test set and stores the latent codes.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 2 — Sensor node: encoding test data")
print("=" * 65)

bytes_original    = INPUT_DIM  * BYTES_PER_FLOAT
bytes_transmitted = LATENT_DIM * BYTES_PER_FLOAT
print(f"\n  Original  : {bytes_original} bytes per array")
print(f"  Compressed: {bytes_transmitted} bytes per array  (ratio {INPUT_DIM // LATENT_DIM}:1)\n")

latent_codes = {}   # name -> np.ndarray of shape (N_TEST, LATENT_DIM)

# AAE encoders
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
    print(f"  {name}: encoded  {X_test.shape} -> {z.shape}")

# PCA encoder
with open(WEIGHTS_DIR / "pca_model.pkl", "rb") as f:
    pca_sensor = pickle.load(f)

z_pca = pca_sensor.transform(X_test_np)
latent_codes["PCA"] = z_pca
print(f"  PCA: encoded  {X_test_np.shape} -> {z_pca.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CLOUD SERVER  (decoder / inverse-transform only)
# Reconstructs the original data from the received latent codes.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PHASE 3 — Cloud server: decoding and reconstruction error")
print("=" * 65)

results = {}   # name -> (mse, mae)

# AAE decoders
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
        x_hat = cloud_decoder(z).cpu().numpy()

    x_hat_original = scaler.inverse_transform(x_hat)
    mse, mae = compute_mse_mae(X_test_np, x_hat)
    results[name] = (mse, mae)
    print(f"\n  {name}  (decoder: {[LATENT_DIM] + decoder_hidden + [INPUT_DIM]})")
    print(f"    MSE: {mse:.6f}  |  MAE: {mae:.6f}")

# PCA decoder
with open(WEIGHTS_DIR / "pca_model.pkl", "rb") as f:
    pca_cloud = pickle.load(f)

x_hat_pca       = pca_cloud.inverse_transform(latent_codes["PCA"])
mse_pca, mae_pca = compute_mse_mae(X_test_np, x_hat_pca)
results["PCA"]  = (mse_pca, mae_pca)
print(f"\n  PCA  (linear baseline)")
print(f"    MSE: {mse_pca:.6f}  |  MAE: {mae_pca:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY  —  all methods, compression 100 -> 25  (4:1)")
print("=" * 65)
print(f"{'Method':<10}  {'MSE':>12}  {'MAE':>12}  {'vs PCA (MSE)':>14}")
mse_pca_ref = results["PCA"][0]
for name, (mse, mae) in results.items():
    delta = mse - mse_pca_ref
    sign  = "+" if delta >= 0 else ""
    ref   = "  (baseline)" if name == "PCA" else f"  {sign}{delta:.6f}"
    print(f"{name:<10}  {mse:>12.6f}  {mae:>12.6f}  {ref}")
