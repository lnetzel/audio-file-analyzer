# Audio File Analyzer

Analyze WAV recordings of oscillators and filters and generate Markdown plans for use as prompts in another application. These utilities describe recorded behavior; they do not build an oscillator or filter.

## Install

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
```

## Oscillator

Pass one WAV file. The report includes recording metadata, signal level, dominant frequency, a waveform estimate, harmonic levels, and a short-time RMS envelope.

```powershell
python Oscillator-analyzer.py "Audio-files/Sine - Clean.wav" --name ms10-sine
```

This writes `oscillator-ms10-sine-plan.md` in the current directory. Use `--output-dir` to choose another output directory.

## Filter

Pass one or more processed recordings. Optionally provide a clean or fully-open recording made with the same stimulus and recording chain as `--reference`.

```powershell
python filter-analyzer.py "Audio-files/White Noise - 50% resonance sweep.wav" --input-kind white-noise --name ms10-noise-sweep
```

With a matching reference:

```powershell
python filter-analyzer.py --reference "path/to/clean-white-noise.wav" --input-kind white-noise --name ms10 "path/to/resonance-sweep.wav"
```

The report writes `filter-[name]-plan.md`. `--input-kind` accepts `auto`, `white-noise`, `tonal`, or `other`. Set it explicitly when the stimulus is known, especially when using white noise.

## Measurement Limits

- A white-noise recording without a matching reference yields an uncalibrated spectrum estimate under a white-noise-input assumption.
- A tone with a moving cutoff yields level-over-time data and, with a matching tone reference, one response point at the tone frequency. It cannot provide a full frequency response.
- A moving-cutoff broadband sweep combines multiple filter settings. It is not a static response curve and does not reveal the knob-to-frequency mapping unless cutoff position is calibrated over time.
- The oscillator analyzer measures the complete recording chain. Filters, effects, gain staging, and the audio interface can all change the reported waveform and harmonics.
- WAVs must have matching sample rates when compared. Record references with the same source, settings, and recording chain as their processed takes.
