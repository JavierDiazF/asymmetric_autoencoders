from dataclasses import dataclass
import numpy as np

@dataclass
class WindowedDataset:
    data: np.ndarray   # ventanas normalizadas (input/target del AE), shape (n, input_dim)
    mins: np.ndarray   # min de cada ventana, shape (n,)
    maxs: np.ndarray   # max de cada ventana, shape (n,)
    refs: np.ndarray   # valor absoluto de referencia de cada ventana, shape (n,)

    def __len__(self):
        return len(self.data)

    def reconstruct(self, index: int, normalized: np.ndarray) -> np.ndarray:
        """Undo normalization + delta for window `index`, given a normalized array
        (can be self.data[index], or the AE output for that same window)."""
        diff = normalized * (self.maxs[index] - self.mins[index]) + self.mins[index]
        return np.cumsum(diff) + self.refs[index]