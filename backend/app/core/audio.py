import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class Condition:
    """Named phone-audio condition and the parameters that define it."""

    name: str
    params: Mapping[str, float | int | bool | None]


CONDITION_REGISTRY: dict[str, Condition] = {
    "clean": Condition("clean", {"phone_path": False}),
    "phone_8k": Condition("phone_8k", {"phone_path": True, "snr_db": None}),
    "phone_snr20": Condition("phone_snr20", {"phone_path": True, "snr_db": 20.0}),
    "phone_snr10": Condition("phone_snr10", {"phone_path": True, "snr_db": 10.0}),
    "phone_snr5": Condition("phone_snr5", {"phone_path": True, "snr_db": 5.0}),
    "phone_snr0": Condition("phone_snr0", {"phone_path": True, "snr_db": 0.0}),
    "phone_clip": Condition(
        "phone_clip",
        {"phone_path": True, "clip_fraction": 0.4, "snr_db": None},
    ),
    "phone_loss10": Condition(
        "phone_loss10",
        {
            "phone_path": True,
            "loss_rate": 0.10,
            "chunk_ms": 20,
            "snr_db": None,
        },
    ),
}

# Short alias for callers that want the registry by its conventional name.
CONDITIONS = CONDITION_REGISTRY


def load_wav(path) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples in [-1, 1], sample_rate)."""
    sample_rate, data = wavfile.read(str(path))

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:  # already float
        audio = data.astype(np.float32)

    if audio.ndim == 2:          # stereo -> mono
        audio = audio.mean(axis=1)

    return audio, sample_rate


def save_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio in [-1, 1] as a 16-bit mono WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Change the sample rate."""
    if sr_from == sr_to:
        return audio
    g = math.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(audio, up, down).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise so that signal-to-noise ratio equals snr_db."""
    signal_power = float(np.mean(audio ** 2))
    if signal_power == 0.0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def hard_clip(audio: np.ndarray, peak_fraction: float) -> np.ndarray:
    """Hard-clip samples at peak_fraction of the input signal's peak amplitude."""
    if not 0.0 <= peak_fraction <= 1.0:
        raise ValueError("peak_fraction must be between 0 and 1")

    output = np.array(audio, copy=True)
    if output.size == 0:
        return output

    peak = float(np.max(np.abs(output)))
    if peak == 0.0:
        return output

    threshold = peak * peak_fraction
    return np.clip(output, -threshold, threshold).astype(output.dtype, copy=False)


def packet_loss(
    audio: np.ndarray,
    sample_rate: int,
    loss_rate: float,
    seed: int = 0,
    chunk_ms: int = 20,
) -> np.ndarray:
    """Zero seeded random chunks totaling approximately the requested fraction."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError("loss_rate must be between 0 and 1")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")

    output = np.array(audio, copy=True)
    if output.size == 0 or loss_rate == 0.0:
        return output

    chunk_samples = max(1, round(sample_rate * chunk_ms / 1000))
    chunk_count = math.ceil(len(output) / chunk_samples)
    rng = np.random.default_rng(seed)
    lost_count = min(chunk_count, round(loss_rate * chunk_count))
    lost = rng.choice(chunk_count, size=lost_count, replace=False)

    for index in lost:
        start = int(index) * chunk_samples
        end = min(start + chunk_samples, len(output))
        output[start:end] = 0

    return output


def mu_law_roundtrip(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Compress with mu-law, quantize to 8 bits, expand back."""
    x = np.clip(audio.astype(np.float64), -1.0, 1.0)

    # compress
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    # quantize to 256 levels (8 bits)
    y_q = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0

    # expand
    out = np.sign(y_q) * ((1.0 + mu) ** np.abs(y_q) - 1.0) / mu
    return out.astype(np.float32)


def telephone_degrade(audio, sr, snr_db=None, seed=0):
    """Make clean audio sound like a phone call. Returns (audio, sample_rate)."""
    original_len = len(audio)

    x = resample(audio, sr, 8000)

    if snr_db is not None:
        x = add_noise(x, snr_db, np.random.default_rng(seed))

    x = mu_law_roundtrip(x)
    x = resample(x, 8000, sr)

    # resampling can be off by a sample or two; force the original length
    if len(x) > original_len:
        x = x[:original_len]
    elif len(x) < original_len:
        x = np.pad(x, (0, original_len - len(x)))

    return x.astype(np.float32), sr


def apply_condition(
    audio: np.ndarray,
    sr: int,
    condition_name: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Apply one registered, deterministic audio condition."""
    try:
        condition = CONDITION_REGISTRY[condition_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown condition {condition_name!r}. "
            f"Valid conditions: {sorted(CONDITION_REGISTRY)}"
        ) from exc

    params = condition.params
    if not params["phone_path"]:
        return audio, sr

    conditioned = audio
    if condition_name == "phone_clip":
        conditioned = hard_clip(conditioned, float(params["clip_fraction"]))
    elif condition_name == "phone_loss10":
        conditioned = packet_loss(
            conditioned,
            sample_rate=sr,
            loss_rate=float(params["loss_rate"]),
            seed=seed,
            chunk_ms=int(params["chunk_ms"]),
        )

    snr_db = params.get("snr_db")
    return telephone_degrade(conditioned, sr, snr_db=snr_db, seed=seed)
