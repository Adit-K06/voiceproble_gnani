import numpy as np
import pytest

from app.core.audio import (
    CONDITION_REGISTRY,
    add_noise,
    apply_condition,
    hard_clip,
    mu_law_roundtrip,
    packet_loss,
    resample,
    telephone_degrade,
)


def sine(sr=16000, secs=1.0, freq=440.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resample_length():
    x = sine(16000)
    y = resample(x, 16000, 8000)
    assert len(y) == 8000


@pytest.mark.parametrize("snr_db", [10.0, 5.0, 0.0])
def test_noise_hits_target_snr(snr_db):
    x = sine()
    noisy = add_noise(x, snr_db, np.random.default_rng(0))
    noise = noisy - x
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean(noise ** 2))
    assert abs(measured - snr_db) < 0.5


def test_noise_is_reproducible():
    x = sine()
    a = add_noise(x, 10, np.random.default_rng(1))
    b = add_noise(x, 10, np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_hard_clip_preserves_length_is_deterministic_and_limits_peak():
    x = sine(amp=0.8)
    clipped = hard_clip(x, peak_fraction=0.6)

    assert len(clipped) == len(x)
    assert np.array_equal(clipped, hard_clip(x, peak_fraction=0.6))
    assert np.max(np.abs(clipped)) <= 0.6 * np.max(np.abs(x)) + 1e-7
    assert np.any(clipped != x)


def test_packet_loss_preserves_length_and_is_deterministic_by_seed():
    x = sine()
    first = packet_loss(x, sample_rate=16000, loss_rate=0.2, seed=23)
    second = packet_loss(x, sample_rate=16000, loss_rate=0.2, seed=23)

    assert len(first) == len(x)
    assert np.array_equal(first, second)
    chunk_samples = round(16000 * 0.020)
    assert any(
        np.all(first[start:start + chunk_samples] == 0)
        for start in range(0, len(first), chunk_samples)
    )


@pytest.mark.parametrize("snr_db", [5.0, 0.0])
def test_low_snr_phone_degradation_preserves_length_and_is_seeded(snr_db):
    x = sine()
    first, first_sr = telephone_degrade(x, 16000, snr_db=snr_db, seed=17)
    second, second_sr = telephone_degrade(x, 16000, snr_db=snr_db, seed=17)

    assert first_sr == second_sr == 16000
    assert len(first) == len(second) == len(x)
    assert np.array_equal(first, second)


def test_mu_law_is_close():
    x = sine()
    y = mu_law_roundtrip(x)
    assert np.max(np.abs(x - y)) < 0.05


def test_degrade_keeps_length_and_rate():
    x = sine()
    y, sr = telephone_degrade(x, 16000, snr_db=15)
    assert sr == 16000
    assert len(y) == len(x)


@pytest.mark.parametrize("condition_name", sorted(CONDITION_REGISTRY))
def test_registered_conditions_preserve_length_and_are_seeded(condition_name):
    sample_rate = 16000
    x = sine(sample_rate, secs=1.0)

    first, first_sr = apply_condition(x, sample_rate, condition_name, seed=42)
    second, second_sr = apply_condition(x, sample_rate, condition_name, seed=42)

    assert first_sr == second_sr == sample_rate
    assert len(first) == len(second) == len(x)
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("condition_name", "target_snr"),
    [
        ("phone_snr20", 20.0),
        ("phone_snr10", 10.0),
        ("phone_snr5", 5.0),
        ("phone_snr0", 0.0),
    ],
)
def test_registered_phone_snr_conditions_hit_target(condition_name, target_snr):
    sample_rate = 16000
    x = sine(sample_rate, secs=1.0)
    clean_phone, _ = apply_condition(x, sample_rate, "phone_8k", seed=42)
    noisy_phone, _ = apply_condition(x, sample_rate, condition_name, seed=42)
    noise = noisy_phone - clean_phone
    measured_snr = 10 * np.log10(
        np.mean(clean_phone ** 2) / np.mean(noise ** 2)
    )

    assert abs(measured_snr - target_snr) <= 0.5


def test_packet_loss_fraction_is_close_to_requested_rate():
    sample_rate = 16000
    x = np.ones(sample_rate * 2, dtype=np.float32)

    lost = packet_loss(x, sample_rate, loss_rate=0.10, seed=42)
    lost_fraction = np.mean(lost == 0)

    assert abs(lost_fraction - 0.10) <= 0.02


def test_phone_clip_reduces_peak_before_phone_path():
    sample_rate = 16000
    x = sine(sample_rate, secs=1.0, amp=0.8)

    clean_phone, _ = apply_condition(x, sample_rate, "phone_8k", seed=42)
    clipped_phone, _ = apply_condition(x, sample_rate, "phone_clip", seed=42)

    assert np.max(np.abs(clipped_phone)) < np.max(np.abs(clean_phone))
