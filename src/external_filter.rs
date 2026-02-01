// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

use super::ChipModel;

/// Maximum mixer DC output level; to be removed if the external
/// filter is turned off: ((wave DC + voice DC)*voices + mixer DC)*volume
/// See voice.cc and filter.cc for an explanation of the values.
const MIXER_DC_6581: i32 = ((((0x800 - 0x380) + 0x800) * 0xff * 3 - 0xfff * 0xff / 18) >> 7) * 0x0f;

// External filter circuit component values.
// The C64 audio output stage uses two STC networks:
//
// 1. Low-pass RC filter: R = 10kOhm, C = 1000pF
//    Cutoff = 1/(2*PI*RC) = 15.9kHz
//
// 2. High-pass (DC blocker): R = 10kOhm, C = 10uF
//    Cutoff = 1/(2*PI*RC) = 1.6Hz
//    (assumes 10kOhm audio equipment input impedance)
//
// A BJT voltage follower (2SC1815) connects these stages but its
// effect requires MHz-level sampling to model accurately.
const R_LP: f64 = 10e3; // 10kOhm
const C_LP: f64 = 1000e-12; // 1000pF
const R_HP: f64 = 10e3; // 10kOhm
const C_HP: f64 = 10e-6; // 10uF

/// Default clock frequency (PAL C64)
const DEFAULT_CLOCK_FREQ: f64 = 985248.0;

/// C64 audio output stage filter.
///
/// Models two STC networks: a ~16kHz low-pass followed by a ~1.6Hz high-pass
/// (DC blocker). Uses simplified RC model instead of full BJT (2SC1815)
/// simulation which would require MHz-level sampling.
#[derive(Clone, Copy)]
pub struct ExternalFilter {
    // Configuration
    enabled: bool,
    mixer_dc: i32,
    /// Low-pass filter coefficient (scaled by 2^7 for fixed-point math)
    lp_coeff: i32,
    /// High-pass filter coefficient (scaled by 2^17 for fixed-point math)
    hp_coeff: i32,
    // Runtime State
    /// Low-pass filter state (internal precision)
    lp_state: i32,
    /// High-pass filter state (internal precision)
    hp_state: i32,
}

/// Calculate RC time constant τ = R × C (seconds)
#[inline]
const fn time_constant(resistance: f64, capacitance: f64) -> f64 {
    resistance * capacitance
}

impl ExternalFilter {
    /// Create an external filter model for the selected SID chip.
    pub fn new(chip_model: ChipModel) -> Self {
        let mixer_dc = match chip_model {
            ChipModel::Mos6581 => MIXER_DC_6581,
            ChipModel::Mos8580 => 0,
        };
        let mut filter = Self {
            enabled: true,
            mixer_dc,
            lp_coeff: 0,
            hp_coeff: 0,
            lp_state: 0,
            hp_state: 0,
        };
        filter.set_clock_frequency(DEFAULT_CLOCK_FREQ);
        filter.reset();
        filter
    }

    /// Enable or disable the external audio filter stage.
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the clock frequency and recalculate filter coefficients.
    ///
    /// Uses first-order IIR filter: α = dt / (dt + τ)
    /// where dt = 1/frequency and τ = RC time constant.
    pub fn set_clock_frequency(&mut self, frequency: f64) {
        let dt = 1.0 / frequency;

        // Low-pass: R = 10kOhm, C = 1000pF, cutoff ≈ 15.9kHz
        let lp_alpha = dt / (dt + time_constant(R_LP, C_LP));
        self.lp_coeff = (lp_alpha * (1 << 7) as f64 + 0.5) as i32;

        // High-pass: R = 10kOhm, C = 10uF, cutoff ≈ 1.6Hz
        let hp_alpha = dt / (dt + time_constant(R_HP, C_HP));
        self.hp_coeff = (hp_alpha * (1 << 17) as f64 + 0.5) as i32;
    }

    /// Clock the filter for one cycle.
    #[inline]
    pub const fn clock(&mut self, vi: i32) {
        if self.enabled {
            // Overflow protection: input can be ~24 bits (3 voices * 20-bit * volume),
            // shifted by 11 would overflow i32. Use i64 for intermediates and
            // saturating_add for state to clamp rather than wrap on overflow.
            let vi_scaled = (vi as i64) << 11;
            let dvlp = ((self.lp_coeff as i64 * (vi_scaled - self.lp_state as i64)) >> 7) as i32;
            let dvhp = ((self.hp_coeff as i64 * (self.lp_state as i64 - self.hp_state as i64))
                >> 17) as i32;
            self.lp_state = self.lp_state.saturating_add(dvlp);
            self.hp_state = self.hp_state.saturating_add(dvhp);
        } else {
            self.lp_state = ((vi as i64 - self.mixer_dc as i64) << 11) as i32;
            self.hp_state = 0;
        }
    }

    /// Clock the filter for multiple cycles with constant input.
    #[inline]
    pub fn clock_delta(&mut self, mut delta: u32, vi: i32) {
        if self.enabled {
            // Overflow protection: see clock() for details
            let vi_scaled = (vi as i64) << 11;
            // Maximum delta for filter stability is approximately 8 cycles
            while delta != 0 {
                let step = delta.min(8) as i64;
                let dvlp = ((self.lp_coeff as i64 * step * (vi_scaled - self.lp_state as i64)) >> 7)
                    as i32;
                let dvhp =
                    ((self.hp_coeff as i64 * step * (self.lp_state as i64 - self.hp_state as i64))
                        >> 17) as i32;
                self.lp_state = self.lp_state.saturating_add(dvlp);
                self.hp_state = self.hp_state.saturating_add(dvhp);
                delta -= step as u32;
            }
        } else {
            self.lp_state = ((vi as i64 - self.mixer_dc as i64) << 11) as i32;
            self.hp_state = 0;
        }
    }

    /// Get the filtered output, scaled back from internal precision.
    #[inline]
    pub const fn output(&self) -> i32 {
        // Output is Vlp - Vhp, scaled back by 11 bits
        ((self.lp_state as i64 - self.hp_state as i64) >> 11) as i32
    }

    /// Reset internal filter state to zero.
    pub const fn reset(&mut self) {
        self.lp_state = 0;
        self.hp_state = 0;
    }
}
