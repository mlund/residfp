// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! EKV transistor model filter for 6581.
//!
//! This module implements a physics-based MOS transistor model for accurate
//! emulation of the 6581 SID filter. The model uses the EKV (Enz-Krummenacher-
//! Vittoz) equations which provide smooth transitions between subthreshold,
//! triode, and saturation modes.
//!
//! The filter consists of two integrator stages (HP and BP) connected in a
//! state-variable filter topology.
//!
//! Memory footprint: ~388KB for lookup tables (generated at runtime).

mod config;
mod integrator;
mod opamp;

use alloc::boxed::Box;

pub use self::config::FilterModelConfig;
use self::integrator::Integrator6581;
use super::filter::{mix_filter_output, route_voices, FilterBehavior};

const MIXER_DC_6581: i32 = (-0xfff * 0xff / 18) >> 7;

/// Physics-based EKV filter for 6581.
///
/// Uses two integrators in a state-variable filter topology with lookup tables
/// for accurate transistor modeling.
pub struct Filter6581Ekv {
    /// Filter model configuration and lookup tables.
    config: Box<FilterModelConfig>,

    /// Highpass integrator stage.
    hp_integrator: Integrator6581,

    /// Bandpass integrator stage.
    bp_integrator: Integrator6581,

    /// Highpass output voltage.
    vhp: i32,

    /// Bandpass output voltage.
    vbp: i32,

    /// Lowpass output voltage.
    vlp: i32,

    /// Non-filtered output voltage.
    vnf: i32,

    // Configuration
    enabled: bool,
    fc: u16,
    filt: u8,
    res: u8,

    // Mode
    voice3_off: bool,
    hp_bp_lp: u8,
    vol: u8,

    // Mixer DC offset
    mixer_dc: i32,
}

impl Clone for Filter6581Ekv {
    fn clone(&self) -> Self {
        // Create a fresh filter - config is regenerated
        Self::new()
    }
}

impl Filter6581Ekv {
    /// Creates a new EKV filter with default configuration.
    pub fn new() -> Self {
        let config = Box::new(FilterModelConfig::new());
        let hp_integrator = Integrator6581::new(&config);
        let bp_integrator = Integrator6581::new(&config);

        let mut filter = Self {
            config,
            hp_integrator,
            bp_integrator,
            vhp: 0,
            vbp: 0,
            vlp: 0,
            vnf: 0,
            enabled: true,
            fc: 0,
            filt: 0,
            res: 0,
            voice3_off: false,
            hp_bp_lp: 0,
            vol: 0,
            mixer_dc: MIXER_DC_6581,
        };
        filter.update_cutoff();
        filter
    }

    /// Sets the filter range for tuning to match specific SID chips.
    ///
    /// Range: 0.0 to 1.0
    /// - 0.0: Lowest cutoff frequencies
    /// - 0.5: Default (20e-6 uCox)
    /// - 1.0: Highest cutoff frequencies
    pub fn set_filter_range(&mut self, adjustment: f64) {
        self.config.set_filter_range(adjustment);
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

    /// Updates the cutoff frequency by setting the Vw voltage on both integrators.
    fn update_cutoff(&mut self) {
        let vw = self.config.get_f0_dac(self.fc as usize);
        self.hp_integrator.set_vw(vw);
        self.bp_integrator.set_vw(vw);
    }

    /// Solves the two integrator stages.
    ///
    /// State-variable filter topology:
    /// - HP integrator: Vbp = solve(Vhp)
    /// - BP integrator: Vlp = solve(Vbp)
    /// - Summer: Vhp = Vbp/Q - Vlp - Vi
    #[inline]
    fn solve_integrators(&mut self, vi: i32) {
        // Solve HP integrator: input is Vhp, output is Vbp
        self.vbp = self.hp_integrator.solve(&self.config, self.vhp);

        // Solve BP integrator: input is Vbp, output is Vlp
        self.vlp = self.bp_integrator.solve(&self.config, self.vbp);

        // Summer: compute new Vhp
        // Vhp = Vbp/Q - Vlp - Vi
        // Q is controlled by resonance: ~res/8 for 6581
        // Higher res = higher Q = more resonance
        let q_div = self.get_q_divisor();
        self.vhp = ((self.vbp * q_div) >> 10) - self.vlp - vi;
    }

    /// Returns the Q divisor based on resonance setting.
    ///
    /// For 6581: 1/Q ~ ~res/8
    #[inline]
    const fn get_q_divisor(&self) -> i32 {
        // Inverted resonance bits give 1/Q
        let inv_res = (!self.res & 0x0f) as i32;
        // Scale to 1024 fixed-point: 1024 * (inv_res/8) = 128 * inv_res
        // Add 1 to avoid division by zero when res=15
        (inv_res * 128) + 1
    }
}

impl FilterBehavior for Filter6581Ekv {
    #[inline]
    fn clock(&mut self, mut voice1: i32, mut voice2: i32, mut voice3: i32, mut ext_in: i32) {
        // Scale each voice down from 20 to 13 bits
        voice1 >>= 7;
        voice2 >>= 7;
        voice3 = if self.voice3_off && self.filt & 0x04 == 0 {
            0
        } else {
            voice3 >> 7
        };
        ext_in >>= 7;

        // Bypass filter if disabled
        if !self.enabled {
            self.vnf = voice1 + voice2 + voice3 + ext_in;
            self.vhp = 0;
            self.vbp = 0;
            self.vlp = 0;
            return;
        }

        // Route voices into or around filter
        let (vi, vnf) = route_voices(self.filt, voice1, voice2, voice3, ext_in);
        self.vnf = vnf;

        // Solve the two integrator stages
        self.solve_integrators(vi);
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
        // Scale each voice down from 20 to 13 bits
        voice1 >>= 7;
        voice2 >>= 7;
        voice3 = if self.voice3_off && self.filt & 0x04 == 0 {
            0
        } else {
            voice3 >> 7
        };
        ext_in >>= 7;

        // Bypass filter if disabled
        if !self.enabled {
            self.vnf = voice1 + voice2 + voice3 + ext_in;
            self.vhp = 0;
            self.vbp = 0;
            self.vlp = 0;
            return;
        }

        // Route voices into or around filter
        let (vi, vnf) = route_voices(self.filt, voice1, voice2, voice3, ext_in);
        self.vnf = vnf;

        // Clock the filter for each cycle
        // The EKV model is accurate per-cycle, so we can't skip cycles
        while delta > 0 {
            self.solve_integrators(vi);
            delta -= 1;
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
        self.hp_integrator.reset();
        self.bp_integrator.reset();
        self.update_cutoff();
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set filter curve parameter (API compatibility with simplified filter).
    ///
    /// For EKV filter, this maps to filter range adjustment.
    fn set_filter_curve(&mut self, curve: f64) {
        // Map curve 0-1 to filter range 0-1
        // curve 0.5 (default) -> range ~0.5 (default uCox)
        self.set_filter_range(curve);
    }

    fn get_filter_curve(&self) -> f64 {
        // Return a nominal value since we can't easily reverse the uCox calculation
        0.5
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
        self.fc = ((value as u16) << 3) & 0x7f8 | self.fc & 0x007;
        self.update_cutoff();
    }

    fn set_fc_lo(&mut self, value: u8) {
        self.fc = self.fc & 0x7f8 | (value as u16) & 0x007;
        self.update_cutoff();
    }

    fn set_mode_vol(&mut self, value: u8) {
        self.voice3_off = value & 0x80 != 0;
        self.hp_bp_lp = (value >> 4) & 0x07;
        self.vol = value & 0x0f;
    }

    fn set_res_filt(&mut self, value: u8) {
        self.res = (value >> 4) & 0x0f;
        self.filt = value & 0x0f;
    }
}

impl Default for Filter6581Ekv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_creates_without_panic() {
        let _filter = Filter6581Ekv::new();
    }

    #[test]
    fn filter_clock_produces_output() {
        let mut filter = Filter6581Ekv::new();

        // Enable all voices through filter
        filter.set_res_filt(0x0f);
        // Enable lowpass output
        filter.set_mode_vol(0x1f);

        // Clock with some input
        for _ in 0..1000 {
            filter.clock(0x7fff << 7, 0, 0, 0);
        }

        // Should produce some output
        let output = filter.output();
        assert!(output != 0, "Filter should produce non-zero output");
    }

    #[test]
    fn filter_reset_clears_state() {
        let mut filter = Filter6581Ekv::new();

        filter.set_fc_hi(0xff);
        filter.set_res_filt(0xff);
        filter.set_mode_vol(0xff);

        for _ in 0..100 {
            filter.clock(0x7fff << 7, 0x7fff << 7, 0x7fff << 7, 0);
        }

        filter.reset();

        assert_eq!(filter.fc, 0);
        assert_eq!(filter.filt, 0);
        assert_eq!(filter.res, 0);
        assert_eq!(filter.vhp, 0);
        assert_eq!(filter.vbp, 0);
        assert_eq!(filter.vlp, 0);
    }

    #[test]
    fn filter_disabled_bypasses() {
        let mut filter = Filter6581Ekv::new();

        filter.set_enabled(false);
        filter.set_res_filt(0x0f);
        filter.set_mode_vol(0x1f);

        filter.clock(0x7fff << 7, 0, 0, 0);

        // When disabled, vhp/vbp/vlp should be zero
        assert_eq!(filter.vhp, 0);
        assert_eq!(filter.vbp, 0);
        assert_eq!(filter.vlp, 0);
    }

    #[test]
    fn filter_voice_routing() {
        let mut filter = Filter6581Ekv::new();

        // No voices through filter
        filter.set_res_filt(0x00);
        filter.clock(1 << 7, 2 << 7, 3 << 7, 4 << 7);
        assert_eq!(filter.vnf, 1 + 2 + 3 + 4);

        // Voice 1 through filter
        filter.set_res_filt(0x01);
        filter.clock(1 << 7, 2 << 7, 3 << 7, 4 << 7);
        assert_eq!(filter.vnf, 2 + 3 + 4);
    }
}
