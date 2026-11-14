import math
import io
import struct
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


def read_wav_tolerant(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV, tolerating Gnani's ``0xFFFFFFFF`` data-size header.

    In that case, the data chunk extends to the end of the byte string; this is
    the only reliable length available when the chunk length is a sentinel.
    The SDK's linear PCM output is 16-bit little-endian PCM.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Invalid RIFF/WAVE audio")

    fmt: tuple[int, int, int] | None = None  # channels, sample rate, block align
    pcm: bytes | None = None
    bits_per_sample: int | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8

        if chunk_id == b"fmt ":
            if chunk_size < 16 or chunk_start + 16 > len(data):
                raise ValueError("Invalid WAV format chunk")
            audio_format, channels, sample_rate, _byte_rate, block_align, bits = (
                struct.unpack_from("<HHIIHH", data, chunk_start)
            )
            if audio_format != 1:
                raise ValueError("Only PCM WAV audio is supported")
            fmt = channels, sample_rate, block_align
            bits_per_sample = bits
        elif chunk_id == b"data":
            actual_size = len(data) - chunk_start if chunk_size == 0xFFFFFFFF else min(
                chunk_size, len(data) - chunk_start
            )
            pcm = data[chunk_start : chunk_start + actual_size]
            # A sentinel length cannot identify following chunks, so its PCM
            # data is treated as the remaining file contents.
            if chunk_size == 0xFFFFFFFF:
                break

        next_offset = chunk_start + chunk_size + (chunk_size & 1)
        if next_offset <= offset or next_offset > len(data):
            break
        offset = next_offset

    if fmt is None or pcm is None or bits_per_sample != 16:
        raise ValueError("WAV must contain 16-bit PCM format and data chunks")
    channels, sample_rate, block_align = fmt
    if channels < 1 or sample_rate < 1 or block_align != channels * 2:
        raise ValueError("Invalid WAV channel or sample-rate metadata")
    usable_size = len(pcm) - (len(pcm) % block_align)
    samples = np.frombuffer(pcm[:usable_size], dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
        audio = samples / 32768.0
    else:
        audio = samples.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False), sample_rate


def normalize_wav_bytes(data: bytes) -> bytes:
    """Return valid mono 16-bit PCM WAV bytes from a supported PCM WAV."""
    audio, sample_rate = read_wav_tolerant(data)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    return buffer.getvalue()


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
