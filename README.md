# residfp-rs

### Overview

This is a fork of `resid-rs` a MOS6581/8580 SID emulator engine with accurate analog circuit modeling.
This project injects the following improvements from the
[`libresidfp`](https://github.com/libsidplayfp/libresidfp) project:

- **DAC nonlinearity model** - R-2R ladder with resistor mismatch and missing termination for 6581
- **EKV filter model** - Physics-based MOS transistor model for accurate 6581 filter (feature: `ekv-filter`)
- **Filter curve adjustment** - 0..1 parameter for tuning to match specific SID chips
- **Soft clipping** - Tanh approximation for 16-bit saturation
- **Floating DAC output** - Tracks DAC fade when no waveform selected
- **Noise LFSR pipeline** - Accurate 2-cycle delay modeling
- **LFSR delay bug** - Rate counter wrap-around behavior
- **External filter** - Frequency-dependent coefficients
- **Two-pass sinc resampler** - Chained FIR filters via intermediate frequency for improved efficiency

### Usage

```rust
use residfp::{Sid, ChipModel, SamplingMethod, clock};

// Create a 6581 SID emulator
let mut sid = Sid::new(ChipModel::Mos6581);
sid.set_sampling_parameters(SamplingMethod::Resample, clock::PAL, 48000)?;

// Configure filter (optional)
sid.set_filter_enabled(true);
sid.set_filter_curve(0.5);  // 0.0 = bright, 1.0 = dark

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

## Credits

- Thanks to Dag Lem for his reSID implementation
- Thanks to the libresidfp team for the floating-point filter models
- Thanks to Daniel Collin for motivating me to put this out and helping out with code optimization
- Commodore folks for building an iconic 8-bit machine
- Rust developers for providing an incredible language to develop in
