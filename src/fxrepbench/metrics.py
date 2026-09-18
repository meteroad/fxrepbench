from __future__ import annotations

from typing import Any, Sequence


class MultiResolutionStftDistance:
    """Per-item L_d used by CounterFX-200 and the accompanying paper."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (120, 240, 50),
        win_lengths: Sequence[int] = (600, 1200, 240),
        epsilon: float = 1e-8,
    ) -> None:
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("STFT resolution lists must have equal lengths")
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.hop_sizes = tuple(int(value) for value in hop_sizes)
        self.win_lengths = tuple(int(value) for value in win_lengths)
        self.epsilon = float(epsilon)
        self._windows: dict[tuple[int, str], Any] = {}

    def _window(self, length: int, device: Any) -> Any:
        import torch

        key = (length, str(device))
        if key not in self._windows:
            self._windows[key] = torch.hann_window(length, device=device)
        return self._windows[key]

    def __call__(
        self,
        candidates: Any,
        target: Any,
        *,
        batch_size: int = 8,
        device: Any = "cpu",
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        device = torch.device(device)
        x_all = torch.as_tensor(candidates).detach().float()
        y = torch.as_tensor(target).detach().float()
        if x_all.ndim == 2:
            x_all = x_all.unsqueeze(0)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        if x_all.ndim != 3 or y.ndim != 3:
            raise ValueError("expected candidate and target tensors shaped [N,C,T] and [1,C,T]")
        if x_all.shape[1] != y.shape[1]:
            raise ValueError("candidate and target channel counts differ")

        length = min(x_all.shape[-1], y.shape[-1])
        x_all = x_all[..., :length]
        y = y[..., :length].to(device)
        target_magnitudes: list[Any] = []
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            spectrum = torch.stft(
                y.reshape(-1, length),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=self._window(win_length, device),
                return_complex=True,
            )
            magnitude = torch.sqrt(
                torch.clamp(
                    spectrum.real.square() + spectrum.imag.square(),
                    min=self.epsilon,
                )
            )
            target_magnitudes.append(
                magnitude.reshape(1, y.shape[1], magnitude.shape[-2], magnitude.shape[-1])
            )

        output: list[Any] = []
        for start in range(0, x_all.shape[0], max(1, int(batch_size))):
            x = x_all[start : start + batch_size].to(device)
            shard_loss = torch.zeros(x.shape[0], device=device)
            for fft_size, hop_size, win_length, target_magnitude in zip(
                self.fft_sizes,
                self.hop_sizes,
                self.win_lengths,
                target_magnitudes,
            ):
                spectrum = torch.stft(
                    x.reshape(-1, length),
                    n_fft=fft_size,
                    hop_length=hop_size,
                    win_length=win_length,
                    window=self._window(win_length, device),
                    return_complex=True,
                )
                magnitude = torch.sqrt(
                    torch.clamp(
                        spectrum.real.square() + spectrum.imag.square(),
                        min=self.epsilon,
                    )
                )
                magnitude = magnitude.reshape(
                    x.shape[0], x.shape[1], magnitude.shape[-2], magnitude.shape[-1]
                )
                target_batch = target_magnitude.expand_as(magnitude)
                spectral_convergence = (
                    torch.linalg.vector_norm(
                        target_batch - magnitude,
                        ord=2,
                        dim=(1, 2, 3),
                    )
                    / torch.linalg.vector_norm(
                        target_batch,
                        ord=2,
                        dim=(1, 2, 3),
                    ).clamp_min(self.epsilon)
                )
                log_magnitude = functional.l1_loss(
                    torch.log(magnitude),
                    torch.log(target_batch),
                    reduction="none",
                ).mean(dim=(1, 2, 3))
                shard_loss += spectral_convergence + log_magnitude
            output.append((shard_loss / len(self.fft_sizes)).cpu())
        return torch.cat(output)


def fx_set_f1(predicted: Sequence[str], target: Sequence[str]) -> float:
    """F1 between unordered effect sets; order and controls are ignored."""

    predicted_set = set(predicted)
    target_set = set(target)
    if not predicted_set and not target_set:
        return 1.0
    if not predicted_set or not target_set:
        return 0.0
    true_positive = len(predicted_set & target_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(target_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
