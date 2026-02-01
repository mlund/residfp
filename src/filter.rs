// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

#![allow(clippy::cast_lossless)]

use core::f64;

use super::data::{SPLINE6581_F0, SPLINE8580_F0};
use super::ChipModel;

/// Common interface for SID filter implementations.
///
/// Both the standard spline-based filter and the EKV transistor model
/// implement this trait, enabling runtime selection between them.
/// Common interface for SID filter implementations.
pub trait FilterBehavior {
    /// Clock the filter for one cycle with voice and external inputs.
    fn clock(&mut self, v1: i32, v2: i32, v3: i32, ext: i32);
    /// Clock the filter for multiple cycles.
    fn clock_delta(&mut self, delta: u32, v1: i32, v2: i32, v3: i32, ext: i32);
    /// Get the current filter output value.
    fn output(&self) -> i32;
    /// Reset filter state to initial values.
    fn reset(&mut self);
    /// Enable or disable the filter (bypasses when disabled).
    fn set_enabled(&mut self, enabled: bool);
    /// Set filter curve/range for tuning to match specific SID chips.
    fn set_filter_curve(&mut self, curve: f64);
    /// Get current filter curve parameter.
    fn get_filter_curve(&self) -> f64;
    // Register access
    /// Filter cutoff low byte.
    fn get_fc_lo(&self) -> u8;
    /// Filter cutoff high byte.
    fn get_fc_hi(&self) -> u8;
    /// Filter resonance/routing register.
    fn get_res_filt(&self) -> u8;
    /// Mode/volume register.
    fn get_mode_vol(&self) -> u8;
    /// Set filter cutoff low byte.
    fn set_fc_lo(&mut self, value: u8);
    /// Set filter cutoff high byte.
    fn set_fc_hi(&mut self, value: u8);
    /// Set resonance/routing register.
    fn set_res_filt(&mut self, value: u8);
    /// Set mode/volume register.
    fn set_mode_vol(&mut self, value: u8);
}

const MIXER_DC: i32 = (-0xfff * 0xff / 18) >> 7;

/// Minimum Q factor (~1/√2, critically damped)
const Q_MIN: f64 = 0.707;

/// Maximum cutoff frequency for 1-cycle filter stability (Hz)
const F0_MAX_1CYCLE: f64 = 16000.0;

/// Maximum cutoff frequency for delta-cycle filter stability (Hz)
const F0_MAX_DELTA: f64 = 4000.0;

/// Fixed-point multiplier for 1MHz clock (2^20 / 1_000_000)
const FIXP_SCALE: f64 = 1.048_576;

/// Routes voices into or around the filter based on the filt register.
///
/// Returns `(filtered_input, non_filtered_output)`.
/// The 16-case match is expanded for performance (avoids bit testing overhead).
#[inline]
pub const fn route_voices(filt: u8, v1: i32, v2: i32, v3: i32, ext: i32) -> (i32, i32) {
    match filt {
        0x0 => (0, v1 + v2 + v3 + ext),
        0x1 => (v1, v2 + v3 + ext),
        0x2 => (v2, v1 + v3 + ext),
        0x3 => (v1 + v2, v3 + ext),
        0x4 => (v3, v1 + v2 + ext),
        0x5 => (v1 + v3, v2 + ext),
        0x6 => (v2 + v3, v1 + ext),
        0x7 => (v1 + v2 + v3, ext),
        0x8 => (ext, v1 + v2 + v3),
        0x9 => (v1 + ext, v2 + v3),
        0xa => (v2 + ext, v1 + v3),
        0xb => (v1 + v2 + ext, v3),
        0xc => (v3 + ext, v1 + v2),
        0xd => (v1 + v3 + ext, v2),
        0xe => (v2 + v3 + ext, v1),
        0xf => (v1 + v2 + v3 + ext, 0),
        _ => (0, v1 + v2 + v3 + ext),
    }
}

/// Mixes filter outputs based on the hp_bp_lp mode register.
///
/// Combines highpass, bandpass, and lowpass outputs according to
/// which filter modes are enabled (bits 0-2 of MODE_VOL register).
#[inline]
pub const fn mix_filter_output(vhp: i32, vbp: i32, vlp: i32, hp_bp_lp: u8) -> i32 {
    match hp_bp_lp {
        0x0 => 0,
        0x1 => vlp,
        0x2 => vbp,
        0x3 => vlp + vbp,
        0x4 => vhp,
        0x5 => vlp + vhp,
        0x6 => vbp + vhp,
        0x7 => vlp + vbp + vhp,
        _ => 0,
    }
}

/// The SID filter is modeled with a two-integrator-loop biquadratic filter,
/// which has been confirmed by Bob Yannes to be the actual circuit used in
/// the SID chip.
///
/// Measurements show that excellent emulation of the SID filter is achieved,
/// except when high resonance is combined with high sustain levels.
/// In this case the SID op-amps are performing less than ideally and are
/// causing some peculiar behavior of the SID filter. This however seems to
/// have more effect on the overall amplitude than on the color of the sound.
///
/// The theory for the filter circuit can be found in "Microelectric Circuits"
/// by Adel S. Sedra and Kenneth C. Smith.
/// The circuit is modeled based on the explanation found there except that
/// an additional inverter is used in the feedback from the bandpass output,
/// allowing the summer op-amp to operate in single-ended mode. This yields
/// inverted filter outputs with levels independent of Q, which corresponds with
/// the results obtained from a real SID.
///
/// We have been able to model the summer and the two integrators of the circuit
/// to form components of an IIR filter.
/// Vhp is the output of the summer, Vbp is the output of the first integrator,
/// and Vlp is the output of the second integrator in the filter circuit.
///
/// According to Bob Yannes, the active stages of the SID filter are not really
/// op-amps. Rather, simple NMOS inverters are used. By biasing an inverter
/// into its region of quasi-linear operation using a feedback resistor from
/// input to output, a MOS inverter can be made to act like an op-amp for
/// small signals centered around the switching threshold.
#[derive(Clone, Copy)]
/// Standard SID multimode filter (6581/8580) with curve tuning.
pub struct Filter {
    // Configuration
    chip_model: ChipModel,
    enabled: bool,
    fc: u16,
    filt: u8,
    res: u8,
    /// Filter curve parameter: 0.0 = bright, 1.0 = dark, default = 0.5
    curve: f64,
    // Mode
    voice3_off: bool,
    hp_bp_lp: u8,
    vol: u8,
    // Runtime State
    /// Highpass integrator state.
    pub vhp: i32,
    /// Bandpass integrator state.
    pub vbp: i32,
    /// Lowpass integrator state.
    pub vlp: i32,
    /// Non-filtered mixer output (pre-filter DC offset removed).
    pub vnf: i32,
    // Cutoff Freq/Res
    mixer_dc: i32,
    q_1024_div: i32,
    w0: i32,
    w0_ceil_1: i32,
    w0_ceil_dt: i32,
    // Cutoff Freq Tables
    f0: &'static [i16; 2048],
}

impl Filter {
    /// Create a filter for the given chip model with default curve.
    pub fn new(chip_model: ChipModel) -> Self {
        let f0 = match chip_model {
            ChipModel::Mos6581 => &SPLINE6581_F0,
            ChipModel::Mos8580 => &SPLINE8580_F0,
        };
        let mut filter = Self {
            chip_model,
            enabled: true,
            fc: 0,
            filt: 0,
            res: 0,
            curve: 0.5,
            voice3_off: false,
            hp_bp_lp: 0,
            vol: 0,
            vhp: 0,
            vbp: 0,
            vlp: 0,
            vnf: 0,
            mixer_dc: MIXER_DC,
            q_1024_div: 0,
            w0: 0,
            w0_ceil_1: 0,
            w0_ceil_dt: 0,
            f0,
        };
        filter.set_q();
        filter.set_w0();
        filter
    }

    /// Returns internal filter state [vhp, vbp, vlp, vnf] for filter switching.
    #[allow(dead_code)]
    pub const fn get_state(&self) -> [i32; 4] {
        [self.vhp, self.vbp, self.vlp, self.vnf]
    }

    /// Sets internal filter state from [vhp, vbp, vlp, vnf] for filter switching.
    #[allow(dead_code)]
    pub const fn set_state(&mut self, state: [i32; 4]) {
        [self.vhp, self.vbp, self.vlp, self.vnf] = state;
    }

    fn set_q(&mut self) {
        // Q is controlled linearly by res. Q has approximate range [Q_MIN, 1.7].
        // As resonance is increased, the filter must be clocked more often to keep
        // stable.

        // The coefficient 1024 is dispensed of later by right-shifting 10 times
        // (2 ^ 10 = 1024).
        self.q_1024_div = (1024.0 / (Q_MIN + 1.0 * self.res as f64 / 15.0)) as i32;
    }

    fn set_w0(&mut self) {
        let base_freq = self.f0[self.fc as usize] as f64;

        // Apply curve adjustment to frequency
        // curve: 0.0 = bright (higher freq), 1.0 = dark (lower freq), 0.5 = neutral
        let adjusted_freq = match self.chip_model {
            ChipModel::Mos6581 => {
                // 6581: Frequency offset, approximately ±15% at extremes
                // Maps curve 0->1 to scale 1.15->0.85
                let scale = 1.15 - 0.30 * self.curve;
                base_freq * scale
            }
            ChipModel::Mos8580 => {
                // 8580: Based on cp parameter range (1.8 at curve=0 to 1.2 at curve=1)
                // cp affects integrator response, equivalent to frequency scaling
                // Maps curve 0->1 to scale 1.2->0.8 (normalized around 1.0 at curve=0.5)
                let scale = 1.2 - 0.4 * self.curve;
                base_freq * scale
            }
        };

        // Multiply with FIXP_SCALE to facilitate division by 1_000_000 by right-
        // shifting 20 times (2 ^ 20 = 1048576).
        self.w0 = (2.0 * f64::consts::PI * adjusted_freq * FIXP_SCALE) as i32;

        // Limit f0 to keep 1-cycle filter stable.
        let w0_max_1 = (2.0 * f64::consts::PI * F0_MAX_1CYCLE * FIXP_SCALE) as i32;
        self.w0_ceil_1 = self.w0.min(w0_max_1);

        // Limit f0 to keep delta-cycle filter stable.
        let w0_max_dt = (2.0 * f64::consts::PI * F0_MAX_DELTA * FIXP_SCALE) as i32;
        self.w0_ceil_dt = self.w0.min(w0_max_dt);
    }
}

impl FilterBehavior for Filter {
    #[inline]
    fn clock(&mut self, mut voice1: i32, mut voice2: i32, mut voice3: i32, mut ext_in: i32) {
        // Scale each voice down from 20 to 13 bits.
        voice1 >>= 7;
        voice2 >>= 7;
        // NB! Voice 3 is not silenced by voice3off if it is routed through
        // the filter.
        voice3 = if self.voice3_off && self.filt & 0x04 == 0 {
            0
        } else {
            voice3 >> 7
        };
        ext_in >>= 7;

        // This is handy for testing.
        if !self.enabled {
            self.vnf = voice1 + voice2 + voice3 + ext_in;
            self.vhp = 0;
            self.vbp = 0;
            self.vlp = 0;
            return;
        }

        let (vi, vnf) = route_voices(self.filt, voice1, voice2, voice3, ext_in);
        self.vnf = vnf;

        // delta_t = 1 is converted to seconds given a 1MHz clock by dividing
        // with 1 000 000.

        // Calculate filter outputs.
        // Vhp = Vbp/Q - Vlp - Vi;
        // dVbp = -w0*Vhp*dt;
        // dVlp = -w0*Vbp*dt;
        let dvbp = (self.w0_ceil_1 * self.vhp) >> 20;
        let dvlp = (self.w0_ceil_1 * self.vbp) >> 20;
        self.vbp -= dvbp;
        self.vlp -= dvlp;
        self.vhp = ((self.vbp * self.q_1024_div) >> 10) - self.vlp - vi;
    }

    #[inline]
    fn clock_delta(
        &mut self,
        mut delta: u32,
        mut voice1: i32,
        mut voice2: i32,
        mut voice3: i32,
        mut ext_in: i32,
    ) {
        // Scale each voice down from 20 to 13 bits.
        voice1 >>= 7;
        voice2 >>= 7;
        if self.voice3_off && self.filt & 0x04 == 0 {
            voice3 = 0;
        } else {
            voice3 >>= 7;
        }
        ext_in >>= 7;
        // Enable filter on/off.
        // This is not really part of SID, but is useful for testing.
        // On slow CPUs it may be necessary to bypass the filter to lower the CPU
        // load.
        if !self.enabled {
            self.vnf = voice1 + voice2 + voice3 + ext_in;
            self.vhp = 0;
            self.vbp = 0;
            self.vlp = 0;
            return;
        }

        let (vi, vnf) = route_voices(self.filt, voice1, voice2, voice3, ext_in);
        self.vnf = vnf;

        // Maximum delta cycles for the filter to work satisfactorily under current
        // cutoff frequency and resonance constraints is approximately 8.
        let mut delta_flt = 8;

        while delta != 0 {
            if delta < delta_flt {
                delta_flt = delta;
            }
            // delta_t is converted to seconds given a 1MHz clock by dividing
            // with 1 000 000. This is done in two operations to avoid integer
            // multiplication overflow.

            // Calculate filter outputs.
            // Vhp = Vbp/Q - Vlp - Vi;
            // dVbp = -w0*Vhp*dt;
            // dVlp = -w0*Vbp*dt;
            let w0_delta_t = (self.w0_ceil_dt * delta_flt as i32) >> 6;
            let dvbp = (w0_delta_t * self.vhp) >> 14;
            let dvlp = (w0_delta_t * self.vbp) >> 14;
            self.vbp -= dvbp;
            self.vlp -= dvlp;
            self.vhp = ((self.vbp * self.q_1024_div) >> 10) - self.vlp - vi;

            delta -= delta_flt;
        }
    }

    #[inline]
    fn output(&self) -> i32 {
        if !self.enabled {
            (self.vnf + self.mixer_dc) * self.vol as i32
        } else {
            let vf = mix_filter_output(self.vhp, self.vbp, self.vlp, self.hp_bp_lp);
            (self.vnf + vf + self.mixer_dc) * self.vol as i32
        }
    }

    fn reset(&mut self) {
        self.fc = 0;
        self.filt = 0;
        self.res = 0;
        self.voice3_off = false;
        self.hp_bp_lp = 0;
        self.vol = 0;
        self.vhp = 0;
        self.vbp = 0;
        self.vlp = 0;
        self.vnf = 0;
        self.set_w0();
        self.set_q();
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set filter curve parameter for tuning to match specific SID chips.
    ///
    /// Range: 0.0 (bright/high frequencies) to 1.0 (dark/low frequencies)
    /// Default: 0.5
    ///
    /// For 6581: Shifts the filter cutoff frequency curve
    /// For 8580: Scales the filter response
    fn set_filter_curve(&mut self, curve: f64) {
        self.curve = curve.clamp(0.0, 1.0);
        self.set_w0();
    }

    fn get_filter_curve(&self) -> f64 {
        self.curve
    }

    fn get_fc_hi(&self) -> u8 {
        (self.fc >> 3) as u8
    }

    fn get_fc_lo(&self) -> u8 {
        (self.fc & 0x007) as u8
    }

    fn get_mode_vol(&self) -> u8 {
        let value = if self.voice3_off { 0x80 } else { 0 };
        value | (self.hp_bp_lp << 4) | (self.vol & 0x0f)
    }

    fn get_res_filt(&self) -> u8 {
        (self.res << 4) | (self.filt & 0x0f)
    }

    fn set_fc_hi(&mut self, value: u8) {
        let result = ((value as u16) << 3) & 0x7f8 | self.fc & 0x007;
        self.fc = result;
        self.set_w0();
    }

    fn set_fc_lo(&mut self, value: u8) {
        let result = self.fc & 0x7f8 | (value as u16) & 0x007;
        self.fc = result;
        self.set_w0();
    }

    fn set_mode_vol(&mut self, value: u8) {
        self.voice3_off = value & 0x80 != 0;
        self.hp_bp_lp = (value >> 4) & 0x07;
        self.vol = value & 0x0f;
    }

    fn set_res_filt(&mut self, value: u8) {
        self.res = (value >> 4) & 0x0f;
        self.filt = value & 0x0f;
        self.set_q();
    }
}
