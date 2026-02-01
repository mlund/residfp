// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! EKV model integrator for 6581 filter.
//!
//! This implements the physics-based MOS transistor model for the integrating
//! op-amp circuit in the 6581 filter. The circuit consists of:
//!
//! ```text
//!                    +---C---+
//!                    |       |
//!      vi --o--Rw--o-o--[A>--o-- vo
//!           |      | vx
//!           +--Rs--+
//! ```
//!
//! Where:
//! - Rw is a voltage-controlled resistor (VCR) setting the cutoff frequency
//! - Rs is a "snake" resistor for DC stability
//! - C is the integration capacitor
//! - A is an inverting op-amp modeled via measured transfer function
//!
//! The EKV (Enz-Krummenacher-Vittoz) model provides smooth transitions between
//! subthreshold, triode, and saturation modes.

use super::config::FilterModelConfig;

/// Integrator for 6581 filter using EKV transistor model.
///
/// This struct holds only the integrator state. The config reference is passed
/// to `solve()` to avoid self-referential lifetime issues.
pub struct Integrator6581 {
    /// Capacitor voltage (charge accumulator), scaled.
    vc: i32,

    /// Op-amp output voltage, normalized.
    vx: u16,

    /// Pre-computed (nVddt - Vw)^2 / 2 for VCR gate voltage calculation.
    n_vddt_vw_2: u32,

    /// W/L ratio for snake resistor (cached from config).
    wl_snake: f64,

    /// Normalized Vdd - Vth (cached from config).
    n_vddt: u16,

    /// Normalized threshold voltage (cached from config).
    n_vt: u16,

    /// Normalized minimum voltage (cached from config).
    n_vmin: u16,
}

impl Integrator6581 {
    /// Creates a new integrator, caching constant values from config.
    pub const fn new(config: &FilterModelConfig) -> Self {
        Self {
            vc: 0,
            vx: 0,
            n_vddt_vw_2: 0,
            wl_snake: config.get_wl_snake(),
            n_vddt: config.get_n_vddt(),
            n_vt: config.get_n_vt(),
            n_vmin: config.get_n_vmin(),
        }
    }

    /// Sets the cutoff frequency via the W control voltage.
    ///
    /// This pre-computes (nVddt - Vw)^2 / 2 for the VCR gate voltage calculation.
    #[inline]
    pub const fn set_vw(&mut self, vw: u16) {
        let diff = self.n_vddt.saturating_sub(vw) as u32;
        self.n_vddt_vw_2 = (diff * diff) >> 1;
    }

    /// Resets the integrator state.
    pub const fn reset(&mut self) {
        self.vc = 0;
        self.vx = 0;
        self.n_vddt_vw_2 = 0;
    }

    /// Solves one step of the integrator, returning the output voltage.
    ///
    /// This implements the EKV model for current through both the snake resistor
    /// (always in triode mode) and the voltage-controlled resistor (VCR).
    ///
    /// # Arguments
    /// * `config` - Filter model configuration with lookup tables
    /// * `vi` - Input voltage (normalized)
    ///
    /// # Returns
    /// Output voltage: vx - (vc >> 14)
    #[inline]
    pub fn solve(&mut self, config: &FilterModelConfig, vi: i32) -> i32 {
        // "Snake" currents for triode mode.
        // The snake resistor has Vg = Vdd, so it's always in triode mode.
        let n_vddt = self.n_vddt as u32;
        let vx = self.vx as u32;

        // Vgst = Vddt - Vx (overdrive voltage at source)
        // Vgdt = Vddt - Vi (overdrive voltage at drain)
        // Note: vi can be negative or > nVddt, so use wrapping subtraction
        let vgst = n_vddt.wrapping_sub(vx);
        let vgdt = n_vddt.wrapping_sub(vi as u32);

        let vgst_2 = vgst.wrapping_mul(vgst);
        let vgdt_2 = vgdt.wrapping_mul(vgdt);

        // Snake current: I = K * W/L * (Vgst^2 - Vgdt^2)
        // Scaled by (1/m)*2^13*m*2^16*m*2^16*2^-15 = m*2^30
        // The subtraction is done in u32 then reinterpreted as i32 (C++ behavior)
        let current_factor = config.get_normalized_current_factor::<13>(self.wl_snake);
        let vgst_2_minus_vgdt_2 = vgst_2.wrapping_sub(vgdt_2) as i32;
        let n_i_snake = (current_factor as i32).wrapping_mul(vgst_2_minus_vgdt_2 >> 15);

        // VCR gate voltage.
        // Vg = Vddt - sqrt(((Vddt - Vw)^2 + Vgdt^2) / 2)
        let vg_arg = (self.n_vddt_vw_2.wrapping_add(vgdt_2 >> 1)) >> 16;
        let vg_arg = (vg_arg as usize).min(65535);
        let n_vg = config.get_vcr_n_vg(vg_arg) as i32;

        // EKV model: kVgt = (Vg - Vt) for VCR
        let k_vgt = n_vg - self.n_vt as i32 - self.n_vmin as i32;

        // VCR voltages for EKV model table lookup.
        // Offset by INT16_MIN to get positive table index.
        let k_vgt_vs = k_vgt
            .wrapping_sub(self.vx as i32)
            .wrapping_sub(i16::MIN as i32);
        let k_vgt_vd = k_vgt.wrapping_sub(vi).wrapping_sub(i16::MIN as i32);

        // Clamp to valid table range
        let k_vgt_vs = k_vgt_vs.clamp(0, 65535) as usize;
        let k_vgt_vd = k_vgt_vd.clamp(0, 65535) as usize;

        // VCR current via EKV model: I = Is * (if - ir)
        // Scaled by m*2^15*2^15 = m*2^30
        let i_f = (config.get_vcr_n_ids_term(k_vgt_vs) as u32) << 15;
        let i_r = (config.get_vcr_n_ids_term(k_vgt_vd) as u32) << 15;
        let n_i_vcr = i_f.wrapping_sub(i_r) as i32;

        // Update capacitor charge.
        // Change in capacitor voltage: vc += n_I_snake + n_I_vcr
        self.vc = self.vc.wrapping_add(n_i_snake.wrapping_add(n_i_vcr));

        // Op-amp transfer function: vx = g(vc)
        // Convert capacitor voltage to table index.
        let vc_idx = (self.vc >> 15).wrapping_sub(i16::MIN as i32);
        let vc_idx = vc_idx.clamp(0, 65535) as usize;
        self.vx = config.get_opamp_rev(vc_idx);

        // Return output voltage: vo = vx - vc/2
        // (vc >> 14 gives vc/2 in the same scale as vx)
        (self.vx as i32).wrapping_sub(self.vc >> 14)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Asserts value is within tolerance of expected, with descriptive message.
    macro_rules! assert_close {
        ($actual:expr, $expected:expr, $tol:expr, $msg:expr) => {
            let diff = ($actual - $expected).abs();
            assert!(
                diff <= $tol,
                "{}: expected ~{}, got {} (diff: {})",
                $msg,
                $expected,
                $actual,
                diff
            );
        };
    }

    #[test]
    fn integrator_solves_without_panic() {
        let config = FilterModelConfig::new();
        let mut integrator = Integrator6581::new(&config);

        // Set some cutoff frequency
        integrator.set_vw(32768);

        // Run several solve steps
        let mut output = 0;
        for i in 0..100 {
            let input = (i * 100) as i32;
            output = integrator.solve(&config, input);
        }

        // Just verify it produces some output
        assert!(output != 0 || integrator.vc != 0);
    }

    #[test]
    fn integrator_reset_clears_state() {
        let config = FilterModelConfig::new();
        let mut integrator = Integrator6581::new(&config);

        integrator.set_vw(32768);
        for _ in 0..100 {
            integrator.solve(&config, 1000);
        }

        integrator.reset();

        assert_eq!(integrator.vc, 0);
        assert_eq!(integrator.vx, 0);
        assert_eq!(integrator.n_vddt_vw_2, 0);
    }

    // =========================================================================
    // C++ libresidfp comparison tests
    // Reference values from libresidfp TestFilterSignal.cpp
    // =========================================================================

    /// Compare integrator step response against C++ reference.
    ///
    /// C++ test: Step from silence(28465) to high(45000) at cycle 10
    /// Vw = dac[1024] = 49664
    ///
    /// C++ output (selected cycles):
    /// cycle 0: output = 4908
    /// cycle 9: output = 4919
    /// cycle 10: output = 3757
    /// cycle 20: output = 2080
    /// cycle 49: output = 1740
    #[test]
    fn compare_cpp_step_response() {
        let config = FilterModelConfig::new();
        let mut integrator = Integrator6581::new(&config);

        // C++ uses dac[1024] = 49664
        let vw = config.get_f0_dac(1024);
        integrator.set_vw(vw);

        // Allow tolerance for floating point differences
        const TOL: i32 = 50;

        let silence = 28465i32;
        let high = 45000i32;

        let mut outputs = Vec::new();
        for i in 0..50 {
            let input = if i < 10 { silence } else { high };
            let output = integrator.solve(&config, input);
            outputs.push(output);
        }

        // Compare key points against C++ reference
        assert_close!(outputs[0], 4908, TOL, "cycle 0");
        assert_close!(outputs[9], 4919, TOL, "cycle 9");
        assert_close!(outputs[10], 3757, TOL, "cycle 10 (first after step)");
        assert_close!(outputs[20], 2080, TOL, "cycle 20");
        assert_close!(outputs[49], 1740, TOL, "cycle 49");
    }

    /// Verify cutoff frequency behavior: higher fc = faster response.
    ///
    /// NOTE: C++ crashes when starting directly with high input due to assertions.
    /// We test relative behavior instead: settle from silence, step to high,
    /// verify higher fc produces faster decay (lower output after same cycles).
    #[test]
    fn cutoff_frequency_relative_behavior() {
        let config = FilterModelConfig::new();
        let silence = 28465i32;
        let high = 45000i32;

        let mut outputs = Vec::new();

        for fc in [256, 512, 1024, 2047] {
            let mut integrator = Integrator6581::new(&config);
            let vw = config.get_f0_dac(fc);
            integrator.set_vw(vw);

            // Settle at silence first (like C++ step response test)
            for _ in 0..20 {
                integrator.solve(&config, silence);
            }

            // Step to high and measure after 20 cycles
            let mut output = 0;
            for _ in 0..20 {
                output = integrator.solve(&config, high);
            }
            outputs.push((fc, output));
        }

        // Verify higher fc produces lower output (faster response to step)
        for i in 1..outputs.len() {
            let (fc_prev, out_prev) = outputs[i - 1];
            let (fc_curr, out_curr) = outputs[i];
            assert!(
                out_curr < out_prev,
                "fc={} should produce lower output than fc={}, but got {} vs {}",
                fc_curr,
                fc_prev,
                out_curr,
                out_prev
            );
        }
    }

    /// Compare sine wave response against C++ reference.
    ///
    /// C++ test: Sine wave with amplitude 8000 around silence (28465)
    /// Vw = dac[512] = 45677
    ///
    /// C++ output (selected cycles):
    /// cycle 0: input=28465, output=5971
    /// cycle 5: input=36465, output=5163
    /// cycle 10: input=28465, output=4859
    /// cycle 50: input=28465, output=4282
    #[test]
    fn compare_cpp_sine_response() {
        use core::f64::consts::PI;

        let config = FilterModelConfig::new();
        let mut integrator = Integrator6581::new(&config);

        let vw = config.get_f0_dac(512);
        integrator.set_vw(vw);

        const TOL: i32 = 50;
        let silence = 28465i32;
        let amplitude = 8000i32;

        let mut outputs = Vec::new();
        for i in 0..100 {
            let phase = i as f64 * 2.0 * PI / 20.0;
            let input = silence + (amplitude as f64 * phase.sin()) as i32;
            let output = integrator.solve(&config, input);
            outputs.push(output);
        }

        // Compare key points against C++ reference
        assert_close!(outputs[0], 5971, TOL, "cycle 0");
        assert_close!(outputs[5], 5163, TOL, "cycle 5");
        assert_close!(outputs[10], 4859, TOL, "cycle 10");
        assert_close!(outputs[50], 4282, TOL, "cycle 50");
    }
}
