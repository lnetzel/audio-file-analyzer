"""Shared WAV loading and signal measurements for the analyzer scripts."""

from dataclasses import dataclass
from pathlib import Path
import re
import wave

import numpy as np


@dataclass
class Audio:
    path: Path
    sample_rate: int
    samples: np.ndarray
    channels: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(path: Path) -> Audio:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            if wav.getcomptype() != "NONE":
                raise ValueError("Compressed WAV files are not supported")
            raw_bytes = wav.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Could not read WAV file {path}: {error}") from error

    if sample_rate <= 0 or frame_count == 0:
        raise ValueError(f"WAV file has no usable samples: {path}")

    if sample_width == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64) / 32768
    elif sample_width == 3:
        bytes_24 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = bytes_24[:, 0] | (bytes_24[:, 1] << 8) | (bytes_24[:, 2] << 16)
        samples = ((values ^ 0x800000) - 0x800000).astype(np.float64) / 8388608
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64) / 2147483648
    else:
        raise ValueError(f"Unsupported PCM sample width ({sample_width} bytes) in {path}")

    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1)

    return Audio(Path(path), int(sample_rate), samples, channels)


def signal_metrics(audio: Audio) -> dict[str, float | int]:
    samples = audio.samples
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return {
        "rms_dbfs": db(rms),
        "peak_dbfs": db(peak),
        "crest_db": db(peak / rms) if rms else float("-inf"),
        "dc_percent": float(np.mean(samples) * 100),
        "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 0.999)),
    }


def db(value: float) -> float:
    return 20 * np.log10(max(float(value), 1e-15))


def spectrum(audio: Audio, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    segment_length = min(nperseg, len(audio.samples))
    stride = max(1, segment_length // 2)
    starts = range(0, max(1, len(audio.samples) - segment_length + 1), stride)
    window = np.hanning(segment_length) if segment_length > 1 else np.ones(1)
    scale = audio.sample_rate * np.sum(np.square(window))
    power_spectra = []
    for start in starts:
        segment = audio.samples[start:start + segment_length]
        if len(segment) != segment_length:
            continue
        transformed = np.fft.rfft((segment - np.mean(segment)) * window)
        power = np.square(np.abs(transformed)) / scale
        if segment_length % 2 == 0:
            power[1:-1] *= 2
        else:
            power[1:] *= 2
        power_spectra.append(power)
    return np.fft.rfftfreq(segment_length, 1 / audio.sample_rate), np.mean(power_spectra, axis=0)


def spectral_flatness(frequencies: np.ndarray, power: np.ndarray) -> float:
    mask = (frequencies >= 30) & (frequencies <= min(18000, frequencies[-1]))
    selected = power[mask]
    if selected.size == 0:
        return 0.0
    selected = np.maximum(selected, 1e-30)
    return float(np.exp(np.mean(np.log(selected))) / np.mean(selected))


def classify_signal(frequencies: np.ndarray, power: np.ndarray) -> tuple[str, float]:
    flatness = spectral_flatness(frequencies, power)
    if flatness >= 0.18:
        return "broadband / noise-like", flatness
    if flatness <= 0.025:
        return "tonal", flatness
    return "mixed or strongly shaped", flatness


def dominant_frequency(audio: Audio) -> float | None:
    samples = audio.samples
    if len(samples) < 16:
        return None
    window = np.hanning(len(samples))
    magnitude = np.abs(np.fft.rfft((samples - np.mean(samples)) * window))
    frequencies = np.fft.rfftfreq(len(samples), 1 / audio.sample_rate)
    mask = (frequencies >= 20) & (frequencies <= min(20000, audio.sample_rate / 2))
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    peak_index = int(indices[np.argmax(magnitude[mask])])
    if peak_index == 0 or peak_index >= len(magnitude) - 1:
        return float(frequencies[peak_index])

    left, center, right = np.log(np.maximum(magnitude[peak_index - 1:peak_index + 2], 1e-30))
    denominator = left - 2 * center + right
    offset = 0.5 * (left - right) / denominator if denominator else 0.0
    return float((peak_index + offset) * audio.sample_rate / len(samples))


def sinusoid_amplitude(samples: np.ndarray, sample_rate: int, frequency: float) -> float:
    if frequency <= 0 or frequency >= sample_rate / 2:
        return 0.0
    window = np.hanning(len(samples))
    times = np.arange(len(samples)) / sample_rate
    projection = np.sum((samples - np.mean(samples)) * window * np.exp(-2j * np.pi * frequency * times))
    return float(2 * abs(projection) / np.sum(window))


def harmonic_levels(audio: Audio, fundamental: float, count: int = 16) -> list[tuple[int, float]]:
    base = sinusoid_amplitude(audio.samples, audio.sample_rate, fundamental)
    if base <= 0:
        return []
    result = [(1, 0.0)]
    for harmonic in range(2, count + 1):
        frequency = fundamental * harmonic
        if frequency >= audio.sample_rate / 2:
            break
        amplitude = sinusoid_amplitude(audio.samples, audio.sample_rate, frequency)
        result.append((harmonic, db(amplitude / base)))
    return result


def waveform_guess(harmonics: list[tuple[int, float]]) -> str:
    levels = dict(harmonics)
    upper_partials = [level for number, level in harmonics if number > 1]
    if len(upper_partials) < 2 or max(upper_partials, default=-100) < -30:
        return "sine-like"
    even = max((levels.get(number, -100) for number in (2, 4, 6, 8)), default=-100)
    odd = max((levels.get(number, -100) for number in (3, 5, 7, 9)), default=-100)
    if even < odd - 15:
        if levels.get(3, -100) < -12 and levels.get(5, -100) < -20:
            return "triangle-like"
        return "square-like"
    if even > -35 and odd > -35:
        return "saw-like"
    return "complex / processed"


def rms_envelope(audio: Audio, window_seconds: float = 0.1) -> list[tuple[float, float, float]]:
    width = max(1, int(audio.sample_rate * window_seconds))
    rows = []
    for start in range(0, len(audio.samples), width):
        end = min(start + width, len(audio.samples))
        rms = float(np.sqrt(np.mean(np.square(audio.samples[start:end]))))
        rows.append((start / audio.sample_rate, end / audio.sample_rate, db(rms)))
    return rows


def octave_bands(
    frequencies: np.ndarray,
    output_power: np.ndarray,
    reference_power: np.ndarray | None = None,
    fraction: int = 6,
) -> list[tuple[float, float]]:
    nyquist = frequencies[-1]
    step = 2 ** (1 / fraction)
    centers = []
    center = 20.0
    while center <= nyquist:
        centers.append(center)
        center *= step

    points = []
    half_band = 2 ** (1 / (2 * fraction))
    for center in centers:
        mask = (frequencies >= center / half_band) & (frequencies < center * half_band)
        if not np.any(mask):
            continue
        output_level = float(np.mean(output_power[mask]))
        if reference_power is None:
            level = db(output_level)
        else:
            reference_level = float(np.mean(reference_power[mask]))
            level = 10 * np.log10(max(output_level, 1e-30) / max(reference_level, 1e-30))
        points.append((center, float(level)))

    if reference_power is None and points:
        maximum = max(level for _, level in points)
        points = [(center, level - maximum) for center, level in points]
    return points


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "analysis"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
