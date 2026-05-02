# residfp-rs

A cycle-exact MOS 6581/8580 SID emulator in Rust. Faithful replication
with good realtime performance, not aimed at exposing chip internal state
or adding effects.

### Lineage

- **reSID** (C++) by Dag Lem — the original cycle-exact emulator.
- **libresidfp** (C++) by Antti Lankila and Leandro Nini — added analog
  filter modeling, DAC nonlinearity, and other accuracy improvements.
- **resid-rs** (Rust) by Sebastian Jastrzebski — the Rust port.

This fork backports [`libresidfp`](https://github.com/libsidplayfp/libresidfp)
improvements not present in the original resid-rs:

- **DAC nonlinearity model** - R-2R ladder with resistor mismatch and missing termination for 6581
 - **EKV filter model** - Physics-based MOS transistor model for accurate 6581 filter (feature: `ekv-filter`, ~400KB tables, higher CPU)
- **Filter curve adjustment** - 0..1 parameter for tuning to match specific SID chips
- **Soft clipping** - Tanh approximation for 16-bit saturation
- **Floating DAC output** - Tracks DAC fade when no waveform selected
- **Noise LFSR pipeline** - Accurate 2-cycle delay modeling
- **LFSR delay bug** - Rate counter wrap-around behavior
- **External filter** - Frequency-dependent coefficients
- **Two-pass sinc resampler** - Chained FIR filters via intermediate frequency for improved efficiency

### Usage

```rust
use residfp::{Sid, SidConfig, ChipModel, SamplingMethod, clock};

// Create a 6581 SID emulator
let mut sid = Sid::from_config(SidConfig {
    chip_model: ChipModel::Mos6581,
    sampling_method: SamplingMethod::Resample,
    clock_freq: clock::PAL,
    sample_freq: 48_000,
});

// Configure filter (optional)
sid.set_filter_enabled(true);
sid.set_filter_curve(0.5);  // 0.0 = dark, 1.0 = bright

// Write to SID registers
sid.write(0x00, 0x00);  // Voice 1 frequency low
sid.write(0x01, 0x10);  // Voice 1 frequency high
sid.write(0x04, 0x11);  // Voice 1 control: gate + triangle

// Generate audio samples
let mut buffer = [0i16; 1024];
while delta > 0 {
    let (samples, next_delta) = sid.sample(delta, &mut buffer, 1);
    // Push samples to audio output...
    delta = next_delta;
}
```

### Components

| Component         | Status      |
|-------------------|-------------|
| Envelope          | Done        |
| ExternalFilter    | Done        |
| Filter            | Done        |
| Sampler           | Done        |
| Spline            | Done        |
| Wave              | Done        |
| Sid               | Done        |

### Changelog

- 0.3 - compliance with the original resid
- 0.4 - full sampler support
- 0.5 - closed performance gap largely due to resampling
- 0.6 - SIMD optimization
- 0.7 - continuous integration and GPLv3
- 0.8 - documentation and api refinements/internal cleanup
- 0.9 - migration to Rust 2018
- 1.0 - no_std support
- 1.1 - more idiomatic implementation, removes interior mutability and improves support for async rust
- 1.2 - API ergonomics: `clock::PAL`/`NTSC` constants, `State` save/restore, standard trait derives

## License

GPLv3, matching the original resid-rs. Compatible with the GPLv2-or-later
upstreams (reSID, libresidfp).

## Credits

- Dag Lem — original reSID C++ implementation
- Antti Lankila — floating-point analog filter modeling in libresidfp
- Leandro Nini — libresidfp maintainer
- Sebastian Jastrzebski — original resid-rs Rust port
- Daniel Collin — code optimization help on early resid-rs
- Commodore for the C64; the Rust community for the language
