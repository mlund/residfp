// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

use super::envelope::State as EnvState;
use super::sampler::{Sampler, SamplingMethod};
use super::synth::{FilterBehavior, Synth};
use super::{ChipModel, SamplingError};

use super::clock;

/// Default clock frequency: PAL C64 (~985 kHz)
const DEFAULT_CLOCK_FREQ: u32 = clock::PAL;
/// Default sample rate: CD quality (44.1 kHz)
const DEFAULT_SAMPLE_FREQ: u32 = 44100;
/// Bus value time-to-live in clock cycles (~8ms decay)
const BUS_VALUE_TTL: u32 = 0x2000;

pub mod reg {
    pub const FREQLO1: u8 = 0x00;
    pub const FREQHI1: u8 = 0x01;
    pub const PWLO1: u8 = 0x02;
    pub const PWHI1: u8 = 0x03;
    pub const CR1: u8 = 0x04;
    pub const AD1: u8 = 0x05;
    pub const SR1: u8 = 0x06;
    pub const FREQLO2: u8 = 0x07;
    pub const FREQHI2: u8 = 0x08;
    pub const PWLO2: u8 = 0x09;
    pub const PWHI2: u8 = 0x0a;
    pub const CR2: u8 = 0x0b;
    pub const AD2: u8 = 0x0c;
    pub const SR2: u8 = 0x0d;
    pub const FREQLO3: u8 = 0x0e;
    pub const FREQHI3: u8 = 0x0f;
    pub const PWLO3: u8 = 0x10;
    pub const PWHI3: u8 = 0x11;
    pub const CR3: u8 = 0x12;
    pub const AD3: u8 = 0x13;
    pub const SR3: u8 = 0x14;
    pub const FCLO: u8 = 0x15;
    pub const FCHI: u8 = 0x16;
    pub const RESFILT: u8 = 0x17;
    pub const MODVOL: u8 = 0x18;
    pub const POTX: u8 = 0x19;
    pub const POTY: u8 = 0x1a;
    pub const OSC3: u8 = 0x1b;
    pub const ENV3: u8 = 0x1c;
}

/// Complete SID chip state for save/restore functionality.
///
/// Contains all register values and internal state needed to exactly
/// reproduce the SID's behavior at a given point in time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct State {
    /// All 32 SID registers ($D400-$D41F).
    pub sid_register: [u8; 32],
    /// Last value written to the data bus.
    pub bus_value: u8,
    /// Cycles until bus value decays to zero.
    pub bus_value_ttl: u32,
    /// External audio input level.
    pub ext_in: i32,
    /// Oscillator accumulators (24-bit, one per voice).
    pub accumulator: [u32; 3],
    /// Noise LFSR shift registers (23-bit, one per voice).
    pub shift_register: [u32; 3],
    /// Envelope generator states (0=Attack, 1=DecaySustain, 2=Release).
    pub envelope_state: [u8; 3],
    /// Current envelope output levels (0-255).
    pub envelope_counter: [u8; 3],
    /// Exponential counter values for envelope curve shaping.
    pub exponential_counter: [u8; 3],
    /// Exponential counter period for current envelope level.
    pub exponential_counter_period: [u8; 3],
    /// Flags indicating envelope is held at zero.
    pub hold_zero: [u8; 3],
    /// Rate counters for envelope timing.
    pub rate_counter: [u16; 3],
    /// Rate counter periods (from ADSR settings).
    pub rate_counter_period: [u16; 3],
}

/// MOS 6581/8580 SID chip emulator.
///
/// The SID (Sound Interface Device) is the legendary sound chip used in the
/// Commodore 64. This emulator accurately reproduces its three voices with
/// waveform generators, envelope generators, and the distinctive analog filter.
///
/// # Example
/// ```ignore
/// use residfp::{Sid, SidConfig, ChipModel, SamplingMethod, clock};
///
/// let config = SidConfig {
///     chip_model: ChipModel::Mos6581,
///     sampling_method: SamplingMethod::Resample,
///     clock_freq: clock::PAL,
///     sample_freq: 48_000,
/// };
/// let mut sid = Sid::from_config(config);
///
/// // Write to SID registers
/// sid.write(0x00, 0x00);  // Voice 1 frequency low
/// sid.write(0x01, 0x10);  // Voice 1 frequency high
/// sid.write(0x04, 0x11);  // Voice 1 control: gate + triangle
///
/// // Generate audio samples
/// let mut buffer = [0i16; 1024];
/// let (samples, remaining) = sid.sample(20000, &mut buffer, 1);
/// ```
#[derive(Clone)]
pub struct Sid {
    // Functional Units
    sampler: Sampler,
    // Runtime State
    bus_value: u8,
    bus_value_ttl: u32,
}

/// Configuration for constructing a [`Sid`].
#[cfg(all(feature = "alloc", feature = "std"))]
#[derive(Clone, Debug)]
pub struct SidConfig {
    /// SID chip model to emulate (default: MOS 6581).
    pub chip_model: ChipModel,
    /// Audio sampling method (default: `SamplingMethod::Fast`).
    pub sampling_method: SamplingMethod,
    /// SID clock frequency in Hz (default: PAL C64 clock).
    pub clock_freq: u32,
    /// Output sample rate in Hz (default: 44.1kHz).
    pub sample_freq: u32,
}

#[cfg(all(feature = "alloc", feature = "std"))]
impl Default for SidConfig {
    fn default() -> Self {
        SidConfig {
            chip_model: ChipModel::default(),
            sampling_method: SamplingMethod::Fast,
            clock_freq: DEFAULT_CLOCK_FREQ,
            sample_freq: DEFAULT_SAMPLE_FREQ,
        }
    }
}

impl Sid {
    /// Construct a SID with default PAL clock, 44.1kHz sample rate, and fast sampling.
    pub fn new(chip_model: ChipModel) -> Self {
        Self::from_config_defaults(
            chip_model,
            SamplingMethod::Fast,
            DEFAULT_CLOCK_FREQ,
            DEFAULT_SAMPLE_FREQ,
        )
    }

    #[inline]
    fn from_config_defaults(
        chip_model: ChipModel,
        sampling_method: SamplingMethod,
        clock_freq: u32,
        sample_freq: u32,
    ) -> Self {
        let synth = Synth::new(chip_model);
        let mut sid = Sid {
            sampler: Sampler::new(synth),
            bus_value: 0,
            bus_value_ttl: 0,
        };
        sid.set_sampling_parameters(sampling_method, clock_freq, sample_freq)
            .expect("default sampling parameters are valid");
        sid
    }

    /// Build a SID instance from a configuration.
    #[cfg(all(feature = "alloc", feature = "std"))]
    /// Construct a SID from a full configuration.
    pub fn from_config(config: SidConfig) -> Self {
        Self::from_config_defaults(
            config.chip_model,
            config.sampling_method,
            config.clock_freq,
            config.sample_freq,
        )
    }

    /// Toggles between standard and EKV transistor model filter.
    ///
    /// The EKV filter uses physics-based MOS transistor modeling for more
    /// accurate 6581 filter behavior, but uses more CPU and memory (~388KB).
    /// Has no effect on 8580 chips (EKV model is 6581-specific).
    ///
    /// Returns `true` if now using EKV filter, `false` if using standard.
    #[cfg(feature = "ekv-filter")]
    pub fn toggle_ekv_filter(&mut self) -> bool {
        self.sampler.synth.toggle_ekv_filter()
    }

    /// Returns `true` if currently using the EKV transistor model filter.
    #[cfg(feature = "ekv-filter")]
    pub fn is_ekv_filter_enabled(&self) -> bool {
        self.sampler.synth.is_ekv_filter_enabled()
    }

    /// Set sampling parameters for audio output.
    ///
    /// # Errors
    /// Returns `SamplingError::ZeroClockFreq` if `clock_freq` is zero.
    /// Returns `SamplingError::ZeroSampleFreq` if `sample_freq` is zero.
    pub fn set_sampling_parameters(
        &mut self,
        method: SamplingMethod,
        clock_freq: u32,
        sample_freq: u32,
    ) -> Result<(), SamplingError> {
        self.sampler
            .set_parameters(method, clock_freq, sample_freq)?;
        // Update external filter coefficients for the new clock frequency
        self.sampler
            .synth
            .ext_filter
            .set_clock_frequency(clock_freq as f64);
        Ok(())
    }

    /// Advance the SID by one clock cycle.
    pub fn clock(&mut self) {
        // Age bus value.
        if self.bus_value_ttl > 0 {
            self.bus_value_ttl -= 1;
            if self.bus_value_ttl == 0 {
                self.bus_value = 0;
            }
        }
        // Clock synthesizer.
        self.sampler.synth.clock();
    }

    /// Advance the SID by `delta` cycles.
    pub fn clock_delta(&mut self, delta: u32) {
        // Age bus value.
        if self.bus_value_ttl >= delta {
            self.bus_value_ttl -= delta;
        } else {
            self.bus_value_ttl = 0;
        }
        if self.bus_value_ttl == 0 {
            self.bus_value = 0;
        }
        // Clock synthesizer.
        self.sampler.synth.clock_delta(delta);
    }

    /// Enable or disable the external output filter (C64 audio stage).
    ///
    /// The external filter models the C64's audio output circuitry:
    /// a low-pass filter (~16kHz) followed by a DC blocking high-pass (~1.6Hz).
    /// Enabled by default.
    pub fn set_external_filter_enabled(&mut self, enabled: bool) {
        self.sampler.synth.ext_filter.set_enabled(enabled);
    }

    /// Enable or disable the internal SID filter.
    ///
    /// The internal filter is the characteristic multimode filter of the SID chip.
    /// Disabling bypasses all filter processing. Enabled by default.
    pub fn set_filter_enabled(&mut self, enabled: bool) {
        self.sampler.synth.filter_impl.set_enabled(enabled);
    }

    /// Set filter curve parameter for tuning to match specific SID chips.
    ///
    /// Range: 0.0 (bright/high frequencies) to 1.0 (dark/low frequencies)
    /// Default: 0.5
    pub fn set_filter_curve(&mut self, curve: f64) {
        self.sampler.synth.filter_impl.set_filter_curve(curve);
    }

    /// Get current filter curve parameter.
    pub fn get_filter_curve(&self) -> f64 {
        self.sampler.synth.filter_impl.get_filter_curve()
    }

    /// Feed an external audio input sample.
    pub fn input(&mut self, sample: i32) {
        // Voice outputs are 20 bits. Scale up to match three voices in order
        // to facilitate simulation of the MOS8580 "digi boost" hardware hack.
        self.sampler.synth.ext_in = (sample << 4) * 3;
    }

    /// Current mixed audio sample (16-bit).
    pub fn output(&self) -> i16 {
        self.sampler.synth.output()
    }

    /// Reset all internal SID state.
    pub fn reset(&mut self) {
        self.sampler.reset();
        self.bus_value = 0;
        self.bus_value_ttl = 0;
    }

    /// SID clocking with audio sampling.
    /// Fixpoint arithmetics is used.
    ///
    /// The example below shows how to clock the SID a specified amount of cycles
    /// while producing audio output:
    /// ``` ignore,
    /// let mut buffer = [0i16; 8192];
    /// while delta > 0 {
    ///     let (samples, next_delta) = self.resid.sample(delta, &mut buffer[..], 1);
    ///     let mut output = self.sound_buffer.lock().unwrap();
    ///     for i in 0..samples {
    ///         output.write(buffer[i]);
    ///     }
    ///     delta = next_delta;
    /// }
    /// ```
    pub fn sample(&mut self, delta: u32, buffer: &mut [i16], interleave: usize) -> (usize, u32) {
        self.sampler.clock(delta, buffer, interleave)
    }

    /// Fill the provided buffer with up to `buffer.len() / interleave` samples.
    ///
    /// This helper produces audio frames without requiring the caller to
    /// compute SID clock deltas; it advances the chip as needed to fill the
    /// buffer and returns the number of frames written.
    ///
    /// Internally uses a large delta, so callers should interleave regular
    /// calls (e.g., once per audio callback) instead of relying on one huge
    /// invocation.
    pub fn sample_frames(&mut self, buffer: &mut [i16], interleave: usize) -> usize {
        let frames = buffer.len() / interleave;
        if frames == 0 {
            return 0;
        }
        let (written, _remaining) = self.sample(u32::MAX, buffer, interleave);
        written
    }

    // -- Device I/O

    /// Read a SID register (applies bus decay).
    pub fn read(&self, reg: u8) -> u8 {
        self.sampler.synth.read(reg, self.bus_value)
    }

    /// Write a SID register.
    pub fn write(&mut self, reg: u8, value: u8) {
        self.bus_value = value;
        self.bus_value_ttl = BUS_VALUE_TTL;
        self.sampler.synth.write(reg, value);
    }

    // -- State

    /// Snapshot full SID state (registers and internals).
    pub fn read_state(&self) -> State {
        let mut state = State {
            sid_register: [0; 32],
            bus_value: 0,
            bus_value_ttl: 0,
            ext_in: 0,
            accumulator: [0; 3],
            shift_register: [0; 3],
            envelope_state: [0; 3],
            envelope_counter: [0; 3],
            exponential_counter: [0; 3],
            exponential_counter_period: [0; 3],
            hold_zero: [0; 3],
            rate_counter: [0; 3],
            rate_counter_period: [0; 3],
        };
        for i in 0..3 {
            let j = i * 7;
            let wave = &self.sampler.synth.voices[i].wave;
            let envelope = &self.sampler.synth.voices[i].envelope;
            state.sid_register[j] = wave.get_frequency_lo();
            state.sid_register[j + 1] = wave.get_frequency_hi();
            state.sid_register[j + 2] = wave.get_pulse_width_lo();
            state.sid_register[j + 3] = wave.get_pulse_width_hi();
            state.sid_register[j + 4] = wave.get_control() | envelope.get_control();
            state.sid_register[j + 5] = envelope.get_attack_decay();
            state.sid_register[j + 6] = envelope.get_sustain_release();
        }
        let filter = &self.sampler.synth.filter_impl;
        state.sid_register[0x15] = filter.get_fc_lo();
        state.sid_register[0x16] = filter.get_fc_hi();
        state.sid_register[0x17] = filter.get_res_filt();
        state.sid_register[0x18] = filter.get_mode_vol();
        for i in 0x19..0x1d {
            state.sid_register[i] = self.read(i as u8);
        }
        for i in 0x1d..0x20 {
            state.sid_register[i] = 0;
        }
        state.bus_value = self.bus_value;
        state.bus_value_ttl = self.bus_value_ttl;
        state.ext_in = self.sampler.synth.ext_in;
        for i in 0..3 {
            let wave = &self.sampler.synth.voices[i].wave;
            let envelope = &self.sampler.synth.voices[i].envelope;
            state.accumulator[i] = wave.get_acc();
            state.shift_register[i] = wave.get_shift();
            state.envelope_state[i] = envelope.state as u8;
            state.envelope_counter[i] = envelope.envelope_counter;
            state.exponential_counter[i] = envelope.exponential_counter;
            state.exponential_counter_period[i] = envelope.exponential_counter_period;
            state.hold_zero[i] = if envelope.hold_zero { 1 } else { 0 };
            state.rate_counter[i] = envelope.rate_counter;
            state.rate_counter_period[i] = envelope.rate_counter_period;
        }
        state
    }

    /// Restore full SID state (registers and internals).
    pub fn write_state(&mut self, state: &State) {
        for i in 0..0x19 {
            self.write(i, state.sid_register[i as usize]);
        }
        self.bus_value = state.bus_value;
        self.bus_value_ttl = state.bus_value_ttl;
        self.sampler.synth.ext_in = state.ext_in;
        for i in 0..3 {
            let envelope = &mut self.sampler.synth.voices[i].envelope;
            self.sampler.synth.voices[i].wave.acc = state.accumulator[i];
            self.sampler.synth.voices[i].wave.shift = state.shift_register[i];
            envelope.state = match state.envelope_state[i] {
                0 => EnvState::Attack,
                1 => EnvState::DecaySustain,
                2 => EnvState::Release,
                _ => EnvState::Release, // Default to Release for invalid states
            };
            envelope.envelope_counter = state.envelope_counter[i];
            envelope.exponential_counter = state.exponential_counter[i];
            envelope.exponential_counter_period = state.exponential_counter_period[i];
            envelope.hold_zero = state.hold_zero[i] != 0;
            envelope.rate_counter = state.rate_counter[i];
            envelope.rate_counter_period = state.rate_counter_period[i];
        }
    }
}
