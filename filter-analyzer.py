"""Analyze filter recordings and write a measurement-based implementation plan."""

import argparse
from pathlib import Path

import numpy as np

from audio_analysis import (
    classify_signal,
    db,
    dominant_frequency,
    load_audio,
    markdown_table,
    octave_bands,
    rms_envelope,
    signal_metrics,
    slug,
    spectrum,
    sinusoid_amplitude,
)


def resonance_candidate(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    levels = [level for _, level in points]
    candidates = []
    for index in range(2, len(points) - 2):
        level = levels[index]
        if level <= max(levels[index - 2:index]) or level <= max(levels[index + 1:index + 3]):
            continue
        surrounding = levels[max(0, index - 6):index - 2] + levels[index + 3:index + 7]
        if not surrounding:
            continue
        prominence = level - float(np.median(surrounding))
        if prominence >= 3:
            candidates.append((points[index][0], level, prominence))
    return max(candidates, key=lambda candidate: candidate[2]) if candidates else None


def analyze_recording(audio, reference, nperseg: int, input_kind: str) -> tuple[str, list[str]]:
    frequencies, power = spectrum(audio, nperseg)
    detected_kind, flatness = classify_signal(frequencies, power)
    reference_data = None
    reference_kind = None
    if reference is not None:
        reference_frequencies, reference_power = spectrum(reference, nperseg)
        if not np.array_equal(frequencies, reference_frequencies):
            raise ValueError("Reference and recording must use matching sample rates and analysis settings.")
        reference_data = (reference_frequencies, reference_power)
        reference_kind, _ = classify_signal(reference_frequencies, reference_power)
    kind = input_kind if input_kind != "auto" else (reference_kind or detected_kind)
    metrics = signal_metrics(audio)
    fundamental = dominant_frequency(audio)
    mode_source = "requested setting" if input_kind != "auto" else ("reference" if reference is not None else "recording")

    details = [
        f"- Duration / format: {audio.duration:.3f} s, {audio.sample_rate:,} Hz, {audio.channels} channel(s).",
        f"- Detected output signal: {detected_kind}; spectral flatness {flatness:.4f}.",
        f"- Stimulus analysis mode: {kind} (based on {mode_source}).",
        f"- Overall RMS / peak: {metrics['rms_dbfs']:.2f} / {metrics['peak_dbfs']:.2f} dBFS; clipped samples: {metrics['clipped_samples']}.",
    ]

    if kind == "white-noise" or "noise-like" in kind:
        reference_power = None
        if reference_data is not None:
            _, reference_power = reference_data
        points = octave_bands(frequencies, power, reference_power)
        if reference is None:
            details.append("- Spectrum: uncalibrated estimate, normalized to its strongest 1/6-octave band under a white-noise-input assumption.")
        else:
            details.append("- Spectrum: output level relative to the supplied fully-open reference, in 1/6-octave bands.")
        peak = resonance_candidate(points)
        if peak:
            details.append(f"- Candidate local resonance peak: {peak[0]:.1f} Hz at {peak[1]:.2f} dB, {peak[2]:.2f} dB above its surrounding bands; this is not a calibrated cutoff frequency.")
        else:
            details.append("- No distinct local resonance peak was resolved in the measured bands.")
        band_rows = [[f"{center:.1f}", f"{level:.2f}"] for center, level in points]
        table = markdown_table(["Center frequency (Hz)", "Level (dB)"], band_rows)
        return table, details

    if kind == "tonal":
        if fundamental is not None:
            details.append(f"- Dominant tone: {fundamental:.2f} Hz.")
        if reference is not None and fundamental is not None:
            reference_frequency = dominant_frequency(reference)
            if reference_frequency and abs(reference_frequency - fundamental) / fundamental <= 0.02:
                output_amplitude = sinusoid_amplitude(audio.samples, audio.sample_rate, fundamental)
                reference_amplitude = sinusoid_amplitude(reference.samples, reference.sample_rate, fundamental)
                gain = db(output_amplitude / max(reference_amplitude, 1e-15))
                details.append(f"- Tone level relative to the reference: {gain:.2f} dB at {fundamental:.2f} Hz. This is one frequency point, not a full response curve.")
            else:
                details.append("- Reference tone does not match the recording closely enough for a valid tone-level comparison.")
        else:
            details.append("- A single-tone cutoff sweep cannot reveal a full frequency response without a matching source reference or a broadband stimulus.")
        envelope = rms_envelope(audio)
        highest_window = max(envelope, key=lambda row: row[2])
        details.append(f"- Highest output window: {highest_window[0]:.2f}-{highest_window[1]:.2f} s at {highest_window[2]:.2f} dBFS; sweep timing is not mapped to cutoff frequency.")
        rows = [[f"{start:.2f}-{end:.2f}", f"{level:.2f}"] for start, end, level in envelope]
        table = markdown_table(["Time (s)", "RMS level (dBFS)"], rows)
        return table, details

    details.append("- A full response curve is not inferred from this mixed or shaped input; provide a matching reference or set --input-kind white-noise for a known broadband-noise stimulus.")
    return "No response curve was inferred.", details


def build_report(name: str, recordings: list, reference, nperseg: int, input_kind: str) -> str:
    mode_source = reference or recordings[0]
    mode_frequencies, mode_power = spectrum(mode_source, nperseg)
    detected_mode, _ = classify_signal(mode_frequencies, mode_power)
    analysis_mode = input_kind if input_kind != "auto" else detected_mode
    broadband_mode = analysis_mode == "white-noise" or "noise-like" in analysis_mode
    sections = [
        f"# Filter Plan: {name}",
        "",
        "## Reference and Method",
    ]
    if reference is None and analysis_mode == "tonal":
        sections.append("No matching tone reference was supplied. The report contains absolute output levels over time only; it cannot calculate gain at the test frequency or a full frequency response.")
    elif reference is None and broadband_mode:
        sections.append("No clean reference was supplied. Broadband spectra are normalized and estimated under a white-noise-input assumption; this is not calibrated response data.")
    elif reference is None:
        sections.append("No clean reference was supplied. This input mode does not support a calibrated response estimate.")
    elif analysis_mode == "tonal":
        sections.append(f"Tone reference recording: `{reference.path}`. Comparisons report the tone component at one frequency and the output level over time; they do not form a broadband response curve.")
    elif broadband_mode:
        sections.append(f"Fully-open reference recording: `{reference.path}`. Reported broadband response is relative to this recording, not an absolute unfiltered response.")
    else:
        sections.append(f"Reference recording: `{reference.path}`. Any spectral comparison is relative to this recording and depends on the selected input-kind.")
    sections.extend([
        "The source captures sweep the cutoff over time. Their aggregate spectra combine multiple cutoff positions and must not be interpreted as a static response curve or calibrated knob-to-frequency mapping. No cutoff positions are fabricated.",
        "",
    ])

    for audio in recordings:
        table, details = analyze_recording(audio, reference, nperseg, input_kind)
        sections.extend([
            f"## Recording: {audio.path.stem}",
            f"Source file: `{audio.path}`",
            "",
            *details,
            "",
            "### Measured Data",
            table,
            "",
        ])

    sections.extend([
        "## Resonance and Calibration Notes",
        "A spectral peak is evidence of a measured emphasis, but these moving-cutoff captures do not by themselves identify the cutoff frequency at each moment or separate resonance from the time-varying sweep. To obtain a calibrated response, record a matching fully-open broadband reference and either hold each cutoff setting steady or provide a known cutoff-versus-time calibration.",
        "",
        "## Prompt for the Other Application",
        "Use the tables above as measurements of the recorded filter behavior. Preserve the measurement conditions and uncertainty in any implementation. Treat data from a moving-cutoff capture as a composite sweep observation, not as a static transfer function. Do not infer a control-to-frequency mapping unless a calibration recording is provided.",
        "",
        f"- Target label: {name}",
        f"- Reference: {reference.path.name if reference else 'none; white-noise assumption used only for broadband recordings'}",
        f"- Recordings analyzed: {len(recordings)}",
        "- Requested behavior: low-pass response with resonance; infer only properties supported by the measured tables.",
    ])
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path, help="one or more filter recordings")
    parser.add_argument("--reference", type=Path, help="matching clean or fully-open reference WAV")
    parser.add_argument("--name", help="filter label used in the report name and title")
    parser.add_argument("--input-kind", choices=("auto", "white-noise", "tonal", "other"), default="auto", help="stimulus type; set explicitly when known")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="directory for the Markdown plan")
    args = parser.parse_args()

    try:
        recordings = [load_audio(path) for path in args.recordings]
        reference = load_audio(args.reference) if args.reference else None
        sample_rates = {audio.sample_rate for audio in recordings}
        if reference:
            sample_rates.add(reference.sample_rate)
        if len(sample_rates) != 1:
            raise ValueError("All recordings and the reference must use the same sample rate.")
        nperseg = min(8192, *(len(audio.samples) for audio in recordings), *( [len(reference.samples)] if reference else []))
        if nperseg < 16:
            raise ValueError("Recordings must contain at least 16 samples.")
        name = args.name or recordings[0].path.stem
        report = build_report(name, recordings, reference, nperseg, args.input_kind)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"filter-{slug(name)}-plan.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
