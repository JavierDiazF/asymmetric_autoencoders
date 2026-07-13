import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchprofile import profile_macs

from autoencoder import AsymmetricAutoencoder
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train
from windoweddataset import WindowedDataset
import split_input_data

# Path constants
RESULTS_DIR = Path("results")
SWEEP_A_CSV = RESULTS_DIR / "sweep_a_results.csv"
SWEEP_B_CSV = RESULTS_DIR / "sweep_b_results.csv"
CSV_PATH_FULL_DATA = Path("Datasets/Caples_Lake_N7_2014_2017.csv")
CSV_PATH_TRAIN = Path("Datasets/Caples_Lake_N7_2014_2017_train.csv")
CSV_PATH_VAL = Path("Datasets/Caples_Lake_N7_2014_2017_val.csv")
CSV_PATH_TEST = Path("Datasets/Caples_Lake_N7_2014_2017_test.csv")

# AEs constants
ACTIVATION_FUNCTION = "elu"
## Sweep A: varying latent dimension
INPUT_DIMS_SWEEP_A = [128, 256, 512, 1024]
LATENT_DIMS_SWEEP_A = [4, 8, 16, 32, 64]
HIDDEN_LAYERS_SWEEP_A = 2

## Sweep B: varying number of hidden layers
INPUT_DIM_SWEEP_B = 100
LATENT_DIM_SWEEP_B = 25
HIDDEN_LAYERS_SWEEP_B = [1, 2, 3, 4, 5]

# Train parameters
EPOCHS = 300
PATIENCE = 15
TRAIN_STRIDE = [10, 27, 33] # Step between windows made primes to get more data than the stored in csv file
# Data size
#N_TRAIN, N_VAL, N_TEST = 10000, 2000, 2000
# Trained models
TRAIN_MODELS_N = 10

# Device optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_window_starts(length: int, window_size: int, strides: list) -> list:
    """Get start index of all windows"""
    starts = list()
    for stride in strides:
        if stride <= 0:
            stride = window_size
        aux = list(range(1, length, stride))
        while aux[-1] + window_size > length:
            aux.pop()
        starts += aux
    starts = list(dict.fromkeys(starts))
    np.random.shuffle(starts)
    return starts

def load_temperatures(csv_path: Path) -> np.ndarray:
    """Lee la columna de temperatura entera una sola vez."""
    return pd.read_csv(csv_path, usecols=[1], dtype=np.float32).iloc[:, 0].values

def read_window(temps: np.ndarray, start: int, window_size: int, extra_prev: bool = False) -> np.ndarray:
    """Read only CSV rows needed for a window instead of loading the entire file into memory."""
    if extra_prev:
        return temps[start - 2 : start - 2 + window_size + 1]
    return temps[start - 1 : start - 1 + window_size]

def window_differential_data(window: np.ndarray, window_size: int) -> np.ndarray:
    """Compute the difference in temperature between consecutive rows in a window."""
    diff = np.diff(window, prepend=window[0])   # prepend=window[0] -> primer delta = 0, igual que .fillna(0)
    diff = np.nan_to_num(diff, nan=0.0)
    return diff[-window_size:]

def normalize_window(diff_values):
    """Local normalization of window data (MinMax -> [0,1])"""
    d_min = diff_values.min()
    d_max = diff_values.max()
    denom = d_max - d_min
    normalized = (diff_values - d_min) / denom if denom != 0 else np.zeros_like(diff_values)
    return normalized, d_min, d_max

def slice_data(input_data: Path, input_dim: int, stride: list) -> WindowedDataset:
    """Slice the input data to adjust it to the current input dimension."""
    temps = load_temperatures(input_data)
    starts = get_window_starts(len(temps), input_dim, stride)
    differential_data = np.empty((len(starts), input_dim), dtype=np.float32)
    mins = np.empty(len(starts), dtype=np.float32)
    maxs = np.empty(len(starts), dtype=np.float32)
    ref_value_window = np.empty(len(starts), dtype=np.float32) # Initial value of the window to be used for denormalization
    for i, start in enumerate(starts):
        extra_prev = (start != 1) # If start == 1 -> extra_prev=False so we don't read previous row (header)
        window = read_window(temps, start, input_dim, extra_prev=extra_prev)
        ref_value_window[i] = window[0]  # Store the first value of the window
        diff = window_differential_data(window, input_dim)
        #print(f"Window starting at row {start}: \tWindow size: {len(window)}\tDifferential data size: {len(diff)}")
        differential_data[i], mins[i], maxs[i] = normalize_window(diff)

    return WindowedDataset(differential_data, mins, maxs, ref_value_window)

def get_train_val_test_splits(input_dim: int) -> Tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Get data from already splitted CSV files. If CSV data has not been already splited it will be splited"""
    if not CSV_PATH_TRAIN.exists() or not CSV_PATH_VAL.exists() or not CSV_PATH_TEST.exists():
        print("Splitting data into train, validation and test sets...")
        split_input_data.split_input_csv(CSV_PATH_FULL_DATA, CSV_PATH_TRAIN, CSV_PATH_VAL, CSV_PATH_TEST)
    # Now we slice the splited input data
    train_data = slice_data(CSV_PATH_TRAIN, input_dim, TRAIN_STRIDE)
    val_data = slice_data(CSV_PATH_VAL, input_dim, [input_dim]) # No overlap between validation windows, so stride = input_dim
    test_data = slice_data(CSV_PATH_TEST, input_dim, [input_dim]) # No overlap between test windows, so stride = input_dim
    return train_data, val_data, test_data


def build_hidden_layers(input_dim, latent_dim, n_hidden):
    """Return the list of hidden layer sizes for a given input and latent dimension, and number of hidden layers."""
    if n_hidden <= 0:
        return []
    dims = np.geomspace(input_dim, latent_dim, n_hidden + 2)   # incluye los dos extremos
    hidden_dims = [round(d) for d in dims[1:-1]]            # descarta los extremos (input/latent_dim)

    return hidden_dims

def train_one_config(X_train: WindowedDataset, X_val: WindowedDataset, input_dim: int, latent_dim: int, 
                     n_layers: int, asymmetric: bool = False, epochs: int = EPOCHS, patience: int = PATIENCE, seed: int = 0,
                     ) -> AsymmetricAutoencoder:
    """Train one AE configuration and return the results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Select hidden layers based on whether the AE is symmetric or asymmetric
    if not asymmetric: # Symmetric AE
        encoder_hidden_layers = decoder_hidden_layers = n_layers
    else: # Asymmetric AE
        # n_layers = 1 -> encoder = 0, decoder = 1
        # n_layers = 2 -> encoder = 0, decoder = 2
        # n_layers = 3 -> encoder = 1, decoder = 2
        # n_layers = 4 -> encoder = 1, decoder = 3
        encoder_hidden_layers = (n_layers - 1) // 2 if n_layers > 0 else 0
        decoder_hidden_layers = n_layers - encoder_hidden_layers
    
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, 
                      hidden_dims=build_hidden_layers(input_dim, latent_dim, encoder_hidden_layers),
                      activation=ACTIVATION_FUNCTION, latent_activation="selu")
    decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim,
                      hidden_dims=build_hidden_layers(latent_dim, input_dim, decoder_hidden_layers),
                      activation=ACTIVATION_FUNCTION, output_activation="sigmoid")
    
    model = AsymmetricAutoencoder(encoder, decoder).to(DEVICE)  # channel=None: pure compression
    optimizer = Adam(model.parameters(), lr=2e-3)
    cfg = TrainConfig(
        epochs=epochs, loss="mse", early_stopping_patience=patience,
        early_stopping_min_delta=1e-7, log_every=0, device=str(DEVICE),
    )
    train_tensor = torch.tensor(X_train.data, dtype=torch.float32)
    val_tensor = torch.tensor(X_val.data, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=256)

    train(model, train_loader, optimizer, cfg, val_loader)
    # Return the trained model and the number of MACs for encoder and decoder models
    return model

def test_one_config(model: AsymmetricAutoencoder, X_test: WindowedDataset) -> Dict[str, float]:
    """Test one AE configuration and return the results."""
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test.data, dtype=torch.float32).to(DEVICE)
        x_hat, _ = model(test_tensor)
        predicted_normalized = x_hat.cpu().numpy()

    mse_per_window = []
    mae_per_window = []
    for i in range(len(X_test.data)):
        true_absolute = X_test.reconstruct(i, X_test.data[i])
        pred_absolute = X_test.reconstruct(i, predicted_normalized[i])
        mse_per_window.append(np.mean((true_absolute - pred_absolute) ** 2))
        mae_per_window.append(np.mean(np.abs(true_absolute - pred_absolute)))

    return {'mse_mean': np.mean(mse_per_window),
            'mse_std': np.std(mse_per_window),
            'mse_p95': np.percentile(mse_per_window, 95),
            'mae_mean': np.mean(mae_per_window),
            'mae_std': np.std(mae_per_window),}
    
def run_sweep_a() -> None:
    """Run sweep A: varying input and latent dimension for AEs."""
    print("=" * 70)
    print(f"SWEEP A — dimension  (hidden_layers fixed at {HIDDEN_LAYERS_SWEEP_A})")
    print("=" * 70)
    
    rows = list()
    for input_dim in INPUT_DIMS_SWEEP_A:
        # Slicing of input data to adjuts it to current input dimension
        X_train, X_val, X_test = get_train_val_test_splits(input_dim)
        for latent_dim in LATENT_DIMS_SWEEP_A:
            ratio = input_dim / (latent_dim + 3) # The ratio is calculated with the min, max and ref data that is sent each window
            print(f"\nInput dim: {input_dim}, Latent dim: {latent_dim} (ratio {ratio}:1)")
            for seed in range(TRAIN_MODELS_N):
                print(f"Training symmetric model with seed {seed}...")
                model = train_one_config(X_train, X_val, input_dim, latent_dim, HIDDEN_LAYERS_SWEEP_A, asymmetric=False, seed=seed)
                # Evaluate model on test data
                results_dict = test_one_config(model, X_test)
                print(f"Test results: MSE={results_dict['mse_mean']:.6f}, MAE={results_dict['mae_mean']:.6f}")
                rows.append({
                    'symmetric': True,
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'ratio': ratio,
                    'hidden_layers': HIDDEN_LAYERS_SWEEP_A,
                    'seed': seed,
                    **results_dict
                })

                # Now it will run asymmetric models
                print(f"Training asymmetric model with seed {seed}...")
                model = train_one_config(X_train, X_val, input_dim, latent_dim, HIDDEN_LAYERS_SWEEP_A, asymmetric=True, seed=seed)
                # Evaluate model on test data
                results_dict = test_one_config(model, X_test)
                print(f"Test results: MSE={results_dict['mse_mean']:.6f}, MAE={results_dict['mae_mean']:.6f}")
                rows.append({
                    'symmetric': False,
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'ratio': ratio,
                    'hidden_layers': HIDDEN_LAYERS_SWEEP_A,
                    'seed': seed,
                    **results_dict
                })
    # Save obtained results
    save_csv(rows, SWEEP_A_CSV)

def run_sweep_b() -> None:
    """Run sweep B: varying hidden layers for AEs."""
    print("=" * 70)
    print(f"SWEEP B — hidden layers (input_dim and latent_dim fixed at {INPUT_DIM_SWEEP_B}, {LATENT_DIM_SWEEP_B})")
    print("=" * 70)
    
    rows = list()
    ratio = INPUT_DIM_SWEEP_B / (LATENT_DIM_SWEEP_B + 3) # The ratio is calculated with the min, max and ref data that is sent each window
    # Slicing of input data to adjuts it to current input dimension
    X_train, X_val, X_test = get_train_val_test_splits(INPUT_DIM_SWEEP_B)
    print(f"\nInput dim: {INPUT_DIM_SWEEP_B}, Latent dim: {LATENT_DIM_SWEEP_B} (ratio {ratio}:1)")
    for hidden_layers in HIDDEN_LAYERS_SWEEP_B:
        for seed in range(TRAIN_MODELS_N):
            print(f"Training symmetric model with seed {seed}...")
            model = train_one_config(X_train, X_val, INPUT_DIM_SWEEP_B, LATENT_DIM_SWEEP_B, hidden_layers, asymmetric=False, seed=seed)
            # Evaluate model on test data
            results_dict = test_one_config(model, X_test)
            print(f"Test results: MSE={results_dict['mse_mean']:.6f}, MAE={results_dict['mae_mean']:.6f}")
            rows.append({
                'symmetric': True,
                'encoder_macs': profile_macs(model.encoder, torch.randn(1, INPUT_DIM_SWEEP_B).to(DEVICE)),
                'decoder_macs': profile_macs(model.decoder, torch.randn(1, LATENT_DIM_SWEEP_B).to(DEVICE)),
                'encoder_params': sum(p.numel() for p in model.encoder.parameters()),
                'decoder_params': sum(p.numel() for p in model.decoder.parameters()),
                'input_dim': INPUT_DIM_SWEEP_B,
                'latent_dim': LATENT_DIM_SWEEP_B,
                'ratio': ratio,
                'hidden_layers': hidden_layers,
                'seed': seed,
                **results_dict
            })

            # Now it will run asymmetric models
            print(f"Training asymmetric model with seed {seed}...")
            model = train_one_config(X_train, X_val, INPUT_DIM_SWEEP_B, LATENT_DIM_SWEEP_B, hidden_layers, asymmetric=True, seed=seed)
            # Evaluate model on test data
            results_dict = test_one_config(model, X_test)
            print(f"Test results: MSE={results_dict['mse_mean']:.6f}, MAE={results_dict['mae_mean']:.6f}")
            rows.append({
                'symmetric': False,
                'encoder_macs': profile_macs(model.encoder, torch.randn(1, INPUT_DIM_SWEEP_B).to(DEVICE)),
                'decoder_macs': profile_macs(model.decoder, torch.randn(1, LATENT_DIM_SWEEP_B).to(DEVICE)),
                'encoder_params': sum(p.numel() for p in model.encoder.parameters()),
                'decoder_params': sum(p.numel() for p in model.decoder.parameters()),
                'input_dim': INPUT_DIM_SWEEP_B,
                'latent_dim': LATENT_DIM_SWEEP_B,
                'ratio': ratio,
                'hidden_layers': hidden_layers,
                'seed': seed,
                **results_dict
            })
    # Save obtained results
    save_csv(rows, SWEEP_B_CSV)

# ------------------------------ Save data ---------------------------------
def save_csv(rows: List[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(rows)} rows to {path}")

# ------------------------------ Main ---------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    run_sweep_a()
    run_sweep_b()

if __name__ == "__main__":
    main()