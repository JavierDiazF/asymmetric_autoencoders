"""
Physical-layer building blocks for the wireless / optical-fiber examples.

These are intentionally separate from Encoder/Decoder: per the survey
(Alnaseri et al., 2026), the AE-for-communications baseline is
encoder -> power constraint -> channel -> decoder (Fig. 1b / Fig. 2a),
not a plain encoder -> decoder. Keeping them as standalone nn.Module
pieces means the IoT example (no channel) is unaffected, and wireless /
fiber examples can compose whichever pieces they need.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerNormalization(nn.Module):
    """
    Enforces the transmit-power constraint from Sec. III-C-3 of the survey:
    scales each sample so ||w||_2^2 = n, i.e. unit average power per
    real-valued channel symbol. Applied to the encoder output before the
    channel. Deterministic per-sample (no batch statistics), so it behaves
    identically at training and single-sample inference time.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        n = w.shape[-1]
        energy = w.pow(2).sum(dim=-1, keepdim=True)
        return w * torch.sqrt(n / (energy + self.eps))


def add_awgn(y: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Adds white Gaussian noise so the *signal* (not per-symbol) SNR matches snr_db."""
    signal_power = y.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(y) * torch.sqrt(noise_power)
    return y + noise


def sample_snr_db(
    snr_db: float, snr_db_range: Optional[Tuple[float, float]], training: bool
) -> float:
    """
    Shared domain-randomization policy (survey Sec. IV-C): while training,
    draw a fresh SNR uniformly from snr_db_range on every call so the AE
    learns to be robust across a range of channel conditions instead of a
    single operating point; otherwise (eval, or no range given) use snr_db.
    """
    if snr_db_range is not None and training:
        lo, hi = snr_db_range
        return lo + (hi - lo) * torch.rand(1).item()
    return snr_db


class AWGNChannel(nn.Module):
    """
    Additive white Gaussian noise channel — the model-assumed, differentiable
    channel-in-the-loop baseline (Sec. IV-A-1-a). Assumes its input already
    satisfies a unit-average-power constraint (see PowerNormalization).

    If snr_db_range=(lo, hi) is given, a fresh SNR is sampled uniformly at
    random on every forward call *while in training mode*, implementing the
    domain-randomization strategy the survey recommends (Sec. IV-C) so the
    AE is trained to be robust across a range of channel conditions instead
    of overfitting a single operating point. In eval mode, snr_db is used.
    """

    def __init__(self, snr_db: float = 10.0, snr_db_range: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.snr_db = snr_db
        self.snr_db_range = snr_db_range

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        snr_db = sample_snr_db(self.snr_db, self.snr_db_range, self.training)
        return add_awgn(w, snr_db)

    def extra_repr(self) -> str:
        return f"snr_db={self.snr_db}, snr_db_range={self.snr_db_range}"


class OpticalFiberChannel(nn.Module):
    """
    Simplified fiber-channel proxy: a short fixed FIR filter applied along
    the channel-use axis of each I/Q rail (standing in for chromatic-
    dispersion-induced inter-symbol interference) followed by AWGN.

    This is a LINEAR approximation, not a split-step Fourier (SSFM)
    nonlinear fiber model — see survey Sec. V-B: dense/ELU AEs trained over
    a simplified differentiable channel (AWGN+CD) are a reasonable
    compromise for short-block experiments, but do not capture Kerr
    nonlinearity, and feedforward AEs do not capture memory beyond the FIR
    span. For long channel memory the survey points to recurrent (BRNN /
    SBRNN) architectures instead (not implemented here).

    Expects w laid out as [I_0..I_{n-1}, Q_0..Q_{n-1}] (n_symbols per rail),
    matching the C2R convention used in example_wireless.py / example_fiber.py.
    """

    def __init__(
        self,
        n_symbols: int,
        snr_db: float = 15.0,
        snr_db_range: Optional[Tuple[float, float]] = None,
        isi_taps: Optional[List[float]] = None,
    ):
        super().__init__()
        self.n_symbols = n_symbols
        self.snr_db = snr_db
        self.snr_db_range = snr_db_range
        taps = torch.tensor(isi_taps or [0.05, 0.9, 0.05], dtype=torch.float32)
        taps = (taps / taps.sum()).view(1, 1, -1)
        self.register_buffer("taps", taps)

    def _apply_isi(self, rail: torch.Tensor) -> torch.Tensor:
        x = rail.unsqueeze(1)  # (batch, 1, n_symbols)
        pad = self.taps.shape[-1] // 2
        return F.conv1d(x, self.taps, padding=pad).squeeze(1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        expected = 2 * self.n_symbols
        if w.shape[-1] != expected:
            raise ValueError(
                f"OpticalFiberChannel expected last dim {expected} "
                f"(I/Q concat of n_symbols={self.n_symbols}), got {w.shape[-1]}"
            )
        i_rail, q_rail = w[:, : self.n_symbols], w[:, self.n_symbols :]
        i_rail, q_rail = self._apply_isi(i_rail), self._apply_isi(q_rail)
        y = torch.cat([i_rail, q_rail], dim=-1)
        snr_db = sample_snr_db(self.snr_db, self.snr_db_range, self.training)
        return add_awgn(y, snr_db)

    def extra_repr(self) -> str:
        return (
            f"n_symbols={self.n_symbols}, snr_db={self.snr_db}, "
            f"snr_db_range={self.snr_db_range}, taps={self.taps.numel()}"
        )
