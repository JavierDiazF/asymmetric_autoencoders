"""
Example: AAE with separated encoder (sensor) and decoder (cloud/server).

Workflow:
  1. Train the full AsymmetricAutoencoder on historical data (done offline, once)
  2. Save encoder and decoder weights independently to separate files
  3. Sensor node  -> loads only encoder weights -> compresses data -> transmits z
  4. Cloud server -> loads only decoder weights -> receives z     -> reconstructs x
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from pathlib import Path

from autoencoder import AsymmetricAutoencoder
from encoder import Encoder
from decoder import Decoder
from train import TrainConfig, train


# ── Config ───────────────────────────────────────────────────────────────────

INPUT_DIM  = 100
LATENT_DIM = 25
BYTES_PER_FLOAT = 4

WEIGHTS_DIRECTORY = Path("weights")
WEIGHTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
ENCODER_PATH = WEIGHTS_DIRECTORY / "encoder_weights.pt"
DECODER_PATH = WEIGHTS_DIRECTORY / "decoder_weights.pt"

torch.manual_seed(42)
# Utiliza cuda para utilizar la GPU cuando está disponible
# En el caso de los dispositivos IoT esta instrucicón no hace nada porque no tienen GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Synthetic data (replace with your real sensor data) ──────────────────────

N_TRAIN, N_VAL = 3000, 500

X_train = torch.randn(N_TRAIN, INPUT_DIM)
X_val   = torch.randn(N_VAL,   INPUT_DIM)

train_loader = DataLoader(TensorDataset(X_train), batch_size=50, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val),   batch_size=256)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING (offline, done once on a powerful machine)
# ═══════════════════════════════════════════════════════════════════════════════

encoder = Encoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=[],
    latent_activation="selu",
)

decoder = Decoder(
    latent_dim=LATENT_DIM,
    output_dim=INPUT_DIM,
    hidden_dims=[50, 75],        # AAE-2 from the paper
    activation="selu",
    output_activation="sigmoid",
)

model = AsymmetricAutoencoder(encoder, decoder).to(DEVICE)

optimizer = Adam(model.parameters(), lr=3e-3)

cfg = TrainConfig(
    epochs=200,
    loss="mse",
    early_stopping_patience=15,
    early_stopping_min_delta=1e-5,
    log_every=10,
    device=str(DEVICE),
)

print("=" * 60)
print("PHASE 1 — Training")
print("=" * 60)
train(model, train_loader, optimizer, cfg, val_loader)


# ── Save encoder and decoder independently ────────────────────────────────────
# state_dict() contains only the weights, not the architecture.
# The architecture is defined by the constructor, so you need to
# recreate it before loading. This is what allows deploying each part separately.

torch.save(model.encoder.state_dict(), ENCODER_PATH)
torch.save(model.decoder.state_dict(), DECODER_PATH)

print(f"\nEncoder weights saved to: {ENCODER_PATH}  ({ENCODER_PATH.stat().st_size} bytes on disk)")
print(f"Decoder weights saved to: {DECODER_PATH}  ({DECODER_PATH.stat().st_size} bytes on disk)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SENSOR NODE (loads only the encoder)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 2 — Sensor node  (encoder only)")
print("=" * 60)

# On the sensor you only instantiate and load the encoder
sensor_encoder = Encoder(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=[],
    latent_activation="selu",
)
sensor_encoder.load_state_dict(torch.load(ENCODER_PATH, weights_only=True))
sensor_encoder.eval()

# Your real sensor readings would go here (one array of INPUT_DIM values)
x_sensor = torch.randn(INPUT_DIM)

with torch.no_grad():
    # El unsqueeze es porque x_sensor es una sola dimensión y el encoder espera al menos una tabla 2D, entonces le añade una dimensión en la posición 0 (por eso de unsqueeze(0)): 100 -> (1,100)
    # Por eso al final se hace el squeeze que le quita la dimensión adicional retronada por el encoder (1,25) -> 25
    z_transmitted = sensor_encoder(x_sensor.unsqueeze(0)).squeeze(0)

bytes_original    = x_sensor.numel()      * BYTES_PER_FLOAT
bytes_transmitted = z_transmitted.numel() * BYTES_PER_FLOAT
ratio             = bytes_original / bytes_transmitted

print(f"Original data  : {x_sensor.shape}  -> {bytes_original} bytes")
print(f"Compressed (z) : {z_transmitted.shape}  -> {bytes_transmitted} bytes")
print(f"Compression ratio: {ratio:.1f}:1")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CLOUD SERVER (loads only the decoder)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 3 — Cloud server  (decoder only)")
print("=" * 60)

# On the server you only instantiate and load the decoder
cloud_decoder = Decoder(
    latent_dim=LATENT_DIM,
    output_dim=INPUT_DIM,
    hidden_dims=[50, 75],
    activation="selu",
    output_activation="sigmoid",
)
cloud_decoder.load_state_dict(torch.load(DECODER_PATH, weights_only=True))
cloud_decoder.eval()

# Receive z from the sensor and reconstruct
with torch.no_grad():
    # El unsqueeze es porque z_transmitted es una sola dimensión y el decoder espera al menos una tabla 2D, entonces le añade una dimensión en la posición 0: 25 -> (1,25100)
    # Por eso al final se hace el squeeze que le quita la dimensión adicional retronada por el encoder (1,100) -> 100
    x_reconstructed = cloud_decoder(z_transmitted.unsqueeze(0)).squeeze(0)

# El .item() es para pasar del objeto torch a float
mse = ((x_reconstructed - x_sensor) ** 2).mean().item()
mae = (x_reconstructed - x_sensor).abs().mean().item()

print(f"Received z     : {z_transmitted.shape}")
print(f"Reconstructed x: {x_reconstructed.shape}")
print(f"\nReconstruction error:")
print(f"  MSE : {mse:.6f}")
print(f"  MAE : {mae:.6f}")
