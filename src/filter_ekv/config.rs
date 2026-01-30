// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! FilterModelConfig for 6581 EKV filter emulation.
//!
//! Contains lookup tables and physical constants for the physics-based MOS
//! transistor model. Tables are generated at runtime from measured op-amp
//! voltage data.
//!
//! # Singleton Usage
//!
//! Table generation takes ~300ms. Use the singleton to initialize once:
//!
//! ```ignore
//! // At startup (once):
//! FilterModelConfig::init(0.5);  // curve: 0.0 (dark) to 1.0 (bright)
//!
//! // Later (fast):
//! let config = FilterModelConfig::global();
//! ```

use alloc::boxed::Box;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::sync::OnceLock;

/// Global singleton for FilterModelConfig (std feature only).
#[cfg(feature = "std")]
static CONFIG: OnceLock<FilterModelConfig> = OnceLock::new();

use super::opamp::{MonotoneSpline, Point};
use crate::dac::build_dac_table;
use crate::ChipModel;

/// Number of bits in the filter cutoff frequency DAC.
const DAC_BITS: usize = 11;

/// Power supplies generate voltages slightly out of spec.
const VOLTAGE_SKEW: f64 = 1.015;

/// Boltzmann constant (J/K).
const K_BOLTZMANN: f64 = 1.380649e-23;

/// Electron charge (C).
const Q_ELECTRON: f64 = 1.602176634e-19;

/// Operating temperature (Celsius).
const TEMPERATURE: f64 = 27.0;

/// Thermal voltage: Ut = kT/q at 27C ~ 26mV
const UT: f64 = K_BOLTZMANN * (TEMPERATURE + 273.15) / Q_ELECTRON;

/// SID 6581 op-amp voltage transfer function.
///
/// Measured on CAP1B/CAP1A on a chip marked MOS 6581R4AR 0687 14.
/// All measured chips have op-amps with output voltages (and thus input
/// voltages) within the range of 0.81V - 10.31V.
const OPAMP_VOLTAGE: [(f64, f64); 33] = [
    (0.81, 10.31), // Approximate start of actual range
    (2.40, 10.31),
    (2.60, 10.30),
    (2.70, 10.29),
    (2.80, 10.26),
    (2.90, 10.17),
    (3.00, 10.04),
    (3.10, 9.83),
    (3.20, 9.58),
    (3.30, 9.32),
    (3.50, 8.69),
    (3.70, 8.00),
    (4.00, 6.89),
    (4.40, 5.21),
    (4.54, 4.54), // Working point (vi = vo)
    (4.60, 4.19),
    (4.80, 3.00),
    (4.90, 2.30), // Change of curvature
    (4.95, 2.03),
    (5.00, 1.88),
    (5.05, 1.77),
    (5.10, 1.69),
    (5.20, 1.58),
    (5.40, 1.44),
    (5.60, 1.33),
    (5.80, 1.26),
    (6.00, 1.21),
    (6.40, 1.12),
    (7.00, 1.02),
    (7.50, 0.97),
    (8.50, 0.89),
    (10.00, 0.81),
    (10.31, 0.81), // Approximate end of actual range
];

/// Configuration and lookup tables for 6581 EKV filter model.
///
/// Memory footprint: ~388KB for lookup tables.
pub struct FilterModelConfig {
    /// VCR gate voltage lookup table (65536 x u16 = 128KB).
    vcr_n_vg: Box<[u16; 65536]>,

    /// VCR current term lookup table (65536 x f64 = 512KB before uCox scaling).
    /// Stored as f64 and multiplied by uCox at runtime for filter range adjustment.
    vcr_n_ids_term: Box<[f64; 65536]>,

    /// Reverse op-amp transfer function (65536 x u16 = 128KB).
    opamp_rev: Box<[u16; 65536]>,

    /// Filter cutoff frequency DAC lookup (2048 x u16 = 4KB).
    f0_dac: Box<[u16; 2048]>,

    /// Fixed-point scale factor: norm * UINT16_MAX.
    n16: f64,

    /// Normalized Vdd - Vth.
    n_vddt: u16,

    /// Normalized threshold voltage.
    n_vt: u16,

    /// Normalized minimum voltage.
    n_vmin: u16,

    /// Transconductance coefficient (adjustable for filter range).
    u_cox: f64,

    /// Current factor coefficient: denorm * (uCox/2 * 1e-6 / C).
    curr_factor_coeff: f64,

    /// Voltage range for denormalization.
    denorm: f64,

    /// W/L ratio for snake resistor.
    wl_snake: f64,
}

impl FilterModelConfig {
    /// Initialize the global singleton with the given filter curve.
    ///
    /// Returns `true` on success, `false` if already initialized.
    ///
    /// # Arguments
    /// * `curve` - Filter curve from 0.0 (dark) to 1.0 (bright)
    #[cfg(feature = "std")]
    pub fn try_init(curve: f64) -> bool {
        CONFIG.set(Self::with_curve(curve)).is_ok()
    }

    /// Initialize the global singleton with the given filter curve.
    ///
    /// # Panics
    /// Panics if already initialized. Prefer `try_init()` for fallible initialization.
    #[cfg(feature = "std")]
    pub fn init(curve: f64) {
        if !Self::try_init(curve) {
            panic!("FilterModelConfig::init() already called");
        }
    }

    /// Returns the global singleton if initialized.
    #[cfg(feature = "std")]
    pub fn try_global() -> Option<&'static FilterModelConfig> {
        CONFIG.get()
    }

    /// Returns the global singleton.
    ///
    /// # Panics
    /// Panics if `init()` was not called. Prefer `try_global()` for fallible access.
    #[cfg(feature = "std")]
    pub fn global() -> &'static FilterModelConfig {
        Self::try_global().expect("FilterModelConfig::init() must be called first")
    }

    /// Creates config with default curve (0.5).
    pub fn new() -> Self {
        Self::with_curve(0.5)
    }

    /// Creates config with specified filter curve.
    ///
    /// # Arguments
    /// * `curve` - 0.0 (bright/high freq) to 1.0 (dark/low freq). Affects DAC zero.
    ///
    /// Note: uCox defaults to 20e-6. Use `set_filter_range()` to adjust it.
    pub fn with_curve(curve: f64) -> Self {
        let curve = curve.clamp(0.0, 1.0);

        // Physical parameters for 6581
        let vdd = 12.0 * VOLTAGE_SKEW;
        let vth = 1.31;
        let vddt = vdd - vth;
        let c = 470e-12; // Capacitor value (F)
        let wl_vcr = 9.0; // W/L for VCR
        let wl_snake = 1.0 / 115.0;

        // Default uCox = 20e-6 (matches C++ libresidfp default)
        // Can be adjusted 1e-6..40e-6 via set_filter_range()
        let u_cox = 20e-6;

        // DAC parameters - dac_zero varies with curve (C++ formula)
        let dac_zero = 6.65 + (1.0 - curve);
        let dac_scale = 2.63;

        // Voltage range from op-amp data
        let vmin = OPAMP_VOLTAGE[0].0;
        let vmax = vddt.max(OPAMP_VOLTAGE[0].1);
        let denorm = vmax - vmin;
        let norm = 1.0 / denorm;
        let n16 = norm * (u16::MAX as f64);

        let curr_factor_coeff = denorm * (u_cox / 2.0 * 1.0e-6 / c);

        // Compute normalized values
        let n_vddt = to_u16(n16 * (vddt - vmin));
        let n_vt = to_u16(n16 * (vth - vmin));
        let n_vmin = to_u16(n16 * vmin);

        // Build op-amp reverse transfer function lookup table
        let opamp_rev = build_opamp_rev_table(n16, vmin);

        // Build VCR gate voltage lookup table
        let vcr_n_vg = build_vcr_n_vg_table(n16, vddt, vmin);

        // Build VCR current term lookup table (without uCox scaling)
        let vcr_n_ids_term = build_vcr_n_ids_term_table(n16, norm, c, wl_vcr);

        // Build filter cutoff frequency DAC table
        let f0_dac = build_f0_dac_table(n16, vmin, dac_zero, dac_scale);

        FilterModelConfig {
            vcr_n_vg,
            vcr_n_ids_term,
            opamp_rev,
            f0_dac,
            n16,
            n_vddt,
            n_vt,
            n_vmin,
            u_cox,
            curr_factor_coeff,
            denorm,
            wl_snake,
        }
    }

    /// Sets the filter range by adjusting uCox.
    ///
    /// Range: 0.0 to 1.0, maps to uCox in [1e-6, 40e-6].
    pub fn set_filter_range(&mut self, adjustment: f64) {
        let adjustment = adjustment.clamp(0.0, 1.0);
        let new_u_cox = (1.0 + 39.0 * adjustment) * 1e-6;

        // Ignore small changes
        if (self.u_cox - new_u_cox).abs() < 1e-12 {
            return;
        }

        self.set_u_cox(new_u_cox);
    }

    fn set_u_cox(&mut self, new_u_cox: f64) {
        self.u_cox = new_u_cox;
        let c = 470e-12;
        self.curr_factor_coeff = self.denorm * (new_u_cox / 2.0 * 1.0e-6 / c);
    }

    /// Returns normalized current factor for given W/L ratio.
    #[inline]
    pub fn get_normalized_current_factor<const N: u32>(&self, wl: f64) -> u16 {
        to_u16((1 << N) as f64 * self.curr_factor_coeff * wl)
    }

    /// Looks up VCR gate voltage.
    #[inline]
    pub fn get_vcr_n_vg(&self, i: usize) -> u16 {
        self.vcr_n_vg[i]
    }

    /// Looks up VCR current term, scaled by current uCox.
    #[inline]
    pub fn get_vcr_n_ids_term(&self, i: usize) -> u16 {
        to_u16(self.vcr_n_ids_term[i] * self.u_cox)
    }

    /// Looks up reverse op-amp transfer function.
    #[inline]
    pub fn get_opamp_rev(&self, i: usize) -> u16 {
        self.opamp_rev[i]
    }

    /// Looks up filter cutoff frequency DAC output.
    #[inline]
    pub fn get_f0_dac(&self, i: usize) -> u16 {
        self.f0_dac[i]
    }

    /// Returns normalized Vdd - Vth.
    #[inline]
    pub fn get_n_vddt(&self) -> u16 {
        self.n_vddt
    }

    /// Returns normalized threshold voltage.
    #[inline]
    pub fn get_n_vt(&self) -> u16 {
        self.n_vt
    }

    /// Returns normalized minimum voltage.
    #[inline]
    pub fn get_n_vmin(&self) -> u16 {
        self.n_vmin
    }

    /// Returns fixed-point scale factor.
    #[inline]
    pub fn get_n16(&self) -> f64 {
        self.n16
    }

    /// Returns W/L ratio for snake resistor.
    #[inline]
    pub fn get_wl_snake(&self) -> f64 {
        self.wl_snake
    }
}

impl Default for FilterModelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts f64 to u16 with rounding.
fn to_u16(x: f64) -> u16 {
    let tmp = (x + 0.5) as i32;
    debug_assert!(tmp >= 0 && tmp <= u16::MAX as i32);
    tmp as u16
}

/// Builds the reverse op-amp transfer function lookup table.
///
/// Maps capacitor voltage to op-amp input voltage using monotone cubic spline
/// interpolation of the measured op-amp voltage data.
fn build_opamp_rev_table(n16: f64, vmin: f64) -> Box<[u16; 65536]> {
    // Scale op-amp voltage data for spline
    let scaled: alloc::vec::Vec<Point> = OPAMP_VOLTAGE
        .iter()
        .map(|&(vi, vo)| Point {
            // x = scaled capacitor voltage: (vi - vo) / 2, shifted to positive range
            x: n16 * (vi - vo) / 2.0 + 32768.0,
            // y = normalized op-amp input voltage
            y: n16 * (vi - vmin),
        })
        .collect();

    let spline = MonotoneSpline::new(&scaled);

    let mut table = Box::new([0u16; 65536]);
    for x in 0..65536 {
        let (y, _dy) = spline.evaluate(x as f64);
        // Clamp negative values (can occur when interpolating outside range)
        table[x] = if y > 0.0 { to_u16(y) } else { 0 };
    }

    table
}

/// Builds the VCR gate voltage lookup table.
///
/// vcr_nVg[i] = nVddt - sqrt(i << 16)
///
/// The table index is right-shifted 16 times to fit in 16 bits.
fn build_vcr_n_vg_table(n16: f64, vddt: f64, vmin: f64) -> Box<[u16; 65536]> {
    let n_vddt = n16 * (vddt - vmin);

    let mut table = Box::new([0u16; 65536]);
    for i in 0..65536u32 {
        // Argument to sqrt is multiplied by (1 << 16) due to table indexing
        let sqrt_val = ((i as u64) << 16) as f64;
        table[i as usize] = to_u16(n_vddt - sqrt_val.sqrt());
    }

    table
}

/// Builds the VCR current term lookup table (without uCox scaling).
///
/// Based on EKV model:
///   Ids = Is * (if - ir)
///   Is = (2 * Ut^2) * W/L
///   if = ln^2(1 + exp((k*(Vg - Vt) - Vs) / (2*Ut)))
///   ir = ln^2(1 + exp((k*(Vg - Vt) - Vd) / (2*Ut)))
fn build_vcr_n_ids_term_table(n16: f64, norm: f64, c: f64, wl_vcr: f64) -> Box<[f64; 65536]> {
    // Moderate inversion characteristic current (without uCox)
    let is = 2.0 * UT * UT * wl_vcr;

    // Normalized current factor for 1 cycle at 1MHz
    let n15 = norm * (i16::MAX as f64);
    let n_is = n15 * 1.0e-6 / c * is;

    let r_n16_2ut = 1.0 / (n16 * 2.0 * UT);

    let mut table = Box::new([0.0f64; 65536]);
    for i in 0..65536 {
        // kVgt_Vx = k*(Vg - Vt) - Vx, offset by INT16_MIN
        let k_vgt_vx = (i as i32) + i16::MIN as i32;
        let log_term = (k_vgt_vx as f64 * r_n16_2ut).exp().ln_1p();
        // Scaled by m * 2^15 (before uCox multiplication)
        table[i] = n_is * log_term * log_term;
    }

    table
}

/// Builds the filter cutoff frequency DAC lookup table.
fn build_f0_dac_table(n16: f64, vmin: f64, dac_zero: f64, dac_scale: f64) -> Box<[u16; 2048]> {
    // Use the existing DAC model to get normalized bit contributions
    let dac_bits = build_dac_table(DAC_BITS, ChipModel::Mos6581);

    let mut table = Box::new([0u16; 2048]);
    for i in 0..(1 << DAC_BITS) {
        // DAC output voltage
        let fcd = dac_bits[i] as f64;
        let voltage = dac_zero + fcd * dac_scale;
        table[i] = to_u16(n16 * (voltage - vmin));
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_creates_tables() {
        let config = FilterModelConfig::new();

        // Verify table sizes are populated
        assert_eq!(config.vcr_n_vg.len(), 65536);
        assert_eq!(config.vcr_n_ids_term.len(), 65536);
        assert_eq!(config.opamp_rev.len(), 65536);
        assert_eq!(config.f0_dac.len(), 2048);
    }

    #[test]
    fn vcr_n_vg_decreases_with_index() {
        let config = FilterModelConfig::new();

        // Gate voltage should decrease as index increases (sqrt grows)
        assert!(config.vcr_n_vg[0] > config.vcr_n_vg[1000]);
        assert!(config.vcr_n_vg[1000] > config.vcr_n_vg[10000]);
    }

    #[test]
    fn opamp_rev_is_monotone() {
        let config = FilterModelConfig::new();

        // Op-amp reverse function should be roughly monotone increasing
        // (may have small deviations at edges)
        let mut increasing_count = 0;
        for i in 1000..64000 {
            if config.opamp_rev[i] >= config.opamp_rev[i - 1] {
                increasing_count += 1;
            }
        }
        // Allow some tolerance for edge effects
        assert!(
            increasing_count > 60000,
            "Op-amp should be mostly monotone: {}/63000",
            increasing_count
        );
    }

    #[test]
    fn filter_range_adjusts_u_cox() {
        let mut config = FilterModelConfig::new();
        let initial_coeff = config.curr_factor_coeff;

        config.set_filter_range(1.0);
        assert!(config.curr_factor_coeff > initial_coeff);

        config.set_filter_range(0.0);
        assert!(config.curr_factor_coeff < initial_coeff);
    }

    // =========================================================================
    // C++ libresidfp comparison tests
    // Reference values from FilterModelConfig6581 with curve=0.5
    // =========================================================================

    /// Asserts value is within tolerance of expected.
    macro_rules! assert_close_u16 {
        ($actual:expr, $expected:expr, $tol:expr, $msg:expr) => {{
            let diff = ($actual as i32 - $expected as i32).unsigned_abs();
            assert!(
                diff <= $tol,
                "{}: expected {}, got {} (diff: {})",
                $msg,
                $expected,
                $actual,
                diff
            );
        }};
    }

    #[test]
    fn compare_cpp_opamp_rev_table() {
        let config = FilterModelConfig::new();

        // C++ reference values from FilterModelConfig6581::getOpampRev(i)
        assert_close_u16!(config.opamp_rev[0], 0, 1, "opamp_rev[0]");
        assert_close_u16!(config.opamp_rev[1000], 0, 1, "opamp_rev[1000]");
        assert_close_u16!(config.opamp_rev[10000], 14384, 10, "opamp_rev[10000]");
        assert_close_u16!(config.opamp_rev[32768], 24299, 10, "opamp_rev[32768]");
        assert_close_u16!(config.opamp_rev[50000], 36472, 10, "opamp_rev[50000]");
        assert_close_u16!(config.opamp_rev[65535], 65159, 10, "opamp_rev[65535]");
    }

    #[test]
    fn compare_cpp_vcr_n_vg_table() {
        let config = FilterModelConfig::new();

        // C++ reference values from FilterModelConfig6581::getVcr_nVg(i)
        assert_close_u16!(config.vcr_n_vg[0], 65535, 1, "vcr_n_vg[0]");
        assert_close_u16!(config.vcr_n_vg[1000], 57440, 10, "vcr_n_vg[1000]");
        assert_close_u16!(config.vcr_n_vg[10000], 39935, 10, "vcr_n_vg[10000]");
        assert_close_u16!(config.vcr_n_vg[50000], 8292, 10, "vcr_n_vg[50000]");
        assert_close_u16!(config.vcr_n_vg[65535], 0, 1, "vcr_n_vg[65535]");
    }

    #[test]
    fn compare_cpp_vcr_n_ids_term_table() {
        let config = FilterModelConfig::new();

        // C++ reference values from FilterModelConfig6581::getVcr_n_Ids_term(i)
        assert_close_u16!(config.get_vcr_n_ids_term(0), 0, 1, "vcr_n_ids_term[0]");
        assert_close_u16!(
            config.get_vcr_n_ids_term(1000),
            0,
            1,
            "vcr_n_ids_term[1000]"
        );
        assert_close_u16!(
            config.get_vcr_n_ids_term(10000),
            0,
            1,
            "vcr_n_ids_term[10000]"
        );
        assert_close_u16!(
            config.get_vcr_n_ids_term(32768),
            1,
            1,
            "vcr_n_ids_term[32768]"
        );
        assert_close_u16!(
            config.get_vcr_n_ids_term(50000),
            4364,
            50,
            "vcr_n_ids_term[50000]"
        );
        assert_close_u16!(
            config.get_vcr_n_ids_term(65535),
            15780,
            100,
            "vcr_n_ids_term[65535]"
        );
    }

    #[test]
    fn compare_cpp_f0_dac_table() {
        let config = FilterModelConfig::new();

        // C++ reference values from FilterModelConfig6581::getDAC(0.5)
        assert_close_u16!(config.f0_dac[0], 41430, 10, "f0_dac[0]");
        assert_close_u16!(config.f0_dac[256], 43620, 10, "f0_dac[256]");
        assert_close_u16!(config.f0_dac[512], 45676, 10, "f0_dac[512]");
        assert_close_u16!(config.f0_dac[1024], 49664, 10, "f0_dac[1024]");
        assert_close_u16!(config.f0_dac[2047], 58434, 10, "f0_dac[2047]");
    }
}
