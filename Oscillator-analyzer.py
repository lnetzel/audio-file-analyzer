"""Analyze one WAV recording and write an oscillator implementation plan."""

import argparse
from pathlib import Path

from audio_analysis import (
    classify_signal,
    dominant_frequency,
    harmonic_levels,
    load_audio,
    markdown_table,
    rms_envelope,
    signal_metrics,
    slug,
    spectrum,
    waveform_guess,
)


def build_report(audio, name: str) -> str:
    frequencies, power = spectrum(audio)
    signal_type, flatness = classify_signal(frequencies, power)
    fundamental = dominant_frequency(audio)
    metrics = signal_metrics(audio)
    harmonics = harmonic_levels(audio, fundamental) if fundamental else []
    guess = waveform_guess(harmonics) if signal_type == "tonal" else "not reliably classifiable from this recording"

    summary_rows = [
        ["Duration", f"{audio.duration:.3f} s"],
        ["Sample rate", f"{audio.sample_rate:,} Hz"],
        ["Channels", str(audio.channels)],
        ["Signal classification", f"{signal_type} (spectral flatness {flatness:.4f})"],
        ["Dominant frequency", f"{fundamental:.2f} Hz" if fundamental else "not detected"],
        ["Waveform estimate", guess],
        ["RMS level", f"{metrics['rms_dbfs']:.2f} dBFS"],
        ["Peak level", f"{metrics['peak_dbfs']:.2f} dBFS"],
        ["Crest factor", f"{metrics['crest_db']:.2f} dB"],
        ["DC offset", f"{metrics['dc_percent']:.4f}% of full scale"],
        ["Clipped samples", str(metrics["clipped_samples"])],
    ]
    harmonic_rows = [[str(number), f"{level:.2f} dBc"] for number, level in harmonics]
    envelope_rows = [[f"{start:.2f}-{end:.2f}", f"{level:.2f} dBFS"] for start, end, level in rms_envelope(audio)]

    sections = [
        f"# Oscillator Plan: {name}",
        "",
        f"Source recording: `{audio.path}`",
        "",
        "## Measured Characteristics",
        markdown_table(["Measurement", "Result"], summary_rows),
        "",
        "## Harmonic Spectrum",
        "Harmonic amplitudes are measured against the dominant-frequency component; values are capture-specific.",
        "",
        markdown_table(["Harmonic", "Relative level"], harmonic_rows) if harmonic_rows else "No reliable harmonic series was detected.",
        "",
        "## Level Over Time",
        markdown_table(["Time (s)", "RMS level"], envelope_rows),
        "",
        "## Interpretation Limits",
        "The report describes the recorded output, not necessarily the oscillator alone. Any filter, phaser, amplifier, room, interface, or gain staging in the recording chain also affects the measurements. The waveform estimate is a heuristic based on harmonic balance.",
        "",
        "## Prompt for the Other Application",
        "Use the following measured recording as the target for an oscillator model. Preserve these measured values as the reference rather than assuming ideal textbook waveforms:",
        "",
        f"- Target label: {name}",
        f"- Dominant frequency: {fundamental:.2f} Hz" if fundamental else "- Dominant frequency: unresolved",
        f"- Recorded waveform estimate: {guess}",
        f"- Harmonic levels: {', '.join(f'{number}x={level:.1f} dBc' for number, level in harmonics) or 'unresolved'}",
        f"- RMS / peak: {metrics['rms_dbfs']:.2f} / {metrics['peak_dbfs']:.2f} dBFS",
        "- Treat the source recording and analysis chain as part of the measurement unless a clean oscillator-only capture is available.",
    ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="one oscillator WAV recording")
    parser.add_argument("--name", help="oscillator label used in the report name and title")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="directory for the Markdown plan")
    args = parser.parse_args()

    try:
        audio = load_audio(args.wav)
    except ValueError as error:
        parser.error(str(error))

    name = args.name or audio.path.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"oscillator-{slug(name)}-plan.md"
    output_path.write_text(build_report(audio, name), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
