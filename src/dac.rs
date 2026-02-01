// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! DAC nonlinearity model for accurate R-2R ladder emulation.
//!
//! The SID DACs are built as R-2R ladders. The 6581 has imperfect resistor
//! matching (2R/R ~ 2.20) and missing termination at bit 0, causing
//! non-monotonic output. The 8580 has proper termination and matched
//! resistors (2R/R = 2.00), producing linear output.

use alloc::vec::Vec;

use super::ChipModel;

/// Models open circuit from missing termination resistor
const R_INFINITY: f64 = 1e6;

impl ChipModel {
    /// MOSFET leakage causes non-zero output even when transistors are "off"
    const fn leakage(self) -> f64 {
        match self {
            Self::Mos6581 => 0.0075,
            Self::Mos8580 => 0.0035,
        }
    }

    /// 6581 has imperfect resistor matching, 8580 is ideal
    const fn r2r_ratio(self) -> f64 {
        match self {
            Self::Mos6581 => 2.20,
            Self::Mos8580 => 2.00,
        }
    }

    /// 6581 DACs lack termination resistor at bit 0
    const fn has_termination(self) -> bool {
        matches!(self, Self::Mos8580)
    }
}

/// Parallel resistance: r1 || r2
const fn parallel(r1: f64, r2: f64) -> f64 {
    (r1 * r2) / (r1 + r2)
}

/// Computes voltage contribution of a single bit in the R-2R ladder.
fn bit_voltage(set_bit: usize, bits: usize, r2: f64, terminated: bool) -> f64 {
    let r = 1.0;
    let mut vn = 1.0;

    // Tail resistance starts at 2R (terminated) or infinity (unterminated)
    let mut rn = if terminated { r2 } else { R_INFINITY };

    // Walk up to set_bit, accumulating parallel resistances
    for _ in 0..set_bit {
        rn = if rn == R_INFINITY {
            r + r2
        } else {
            r + parallel(r2, rn)
        };
    }

    // Source transformation at set_bit
    if rn == R_INFINITY {
        rn = r2;
    } else {
        let rn_par = parallel(r2, rn);
        vn *= rn_par / r2;
        rn = rn_par;
    }

    // Walk from set_bit to MSB, applying voltage dividers
    for _ in (set_bit + 1)..bits {
        rn += r;
        let i = vn / rn;
        rn = parallel(r2, rn);
        vn = rn * i;
    }

    vn
}

/// Builds per-bit voltage contribution table for R-2R ladder DAC.
fn build_dac_bits(bits: usize, chip_model: ChipModel) -> Vec<f64> {
    let r2 = chip_model.r2r_ratio();
    let terminated = chip_model.has_termination();

    let mut dac_bits: Vec<f64> = (0..bits)
        .map(|bit| bit_voltage(bit, bits, r2, terminated))
        .collect();

    let v_sum: f64 = dac_bits.iter().sum();
    for v in &mut dac_bits {
        *v /= v_sum;
    }

    dac_bits
}

/// Computes DAC output for a single input value.
#[allow(dead_code)]
pub fn compute_dac_output(input: u32, bits: usize, chip_model: ChipModel) -> f32 {
    let dac_bits = build_dac_bits(bits, chip_model);
    let leakage = chip_model.leakage();

    let value: f64 = dac_bits
        .iter()
        .enumerate()
        .map(|(i, &bit_value)| {
            let on = (input & (1 << i)) != 0;
            if on {
                bit_value
            } else {
                bit_value * leakage
            }
        })
        .sum();

    value as f32
}

/// Builds complete DAC lookup table for all input values.
pub fn build_dac_table(bits: usize, chip_model: ChipModel) -> Vec<f32> {
    let dac_bits = build_dac_bits(bits, chip_model);
    let leakage = chip_model.leakage();

    (0..(1 << bits))
        .map(|input| {
            let value: f64 = dac_bits
                .iter()
                .enumerate()
                .map(|(i, &bit_value)| {
                    let on = (input & (1 << i)) != 0;
                    if on {
                        bit_value
                    } else {
                        bit_value * leakage
                    }
                })
                .sum();
            value as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 6581 DAC is non-monotonic due to missing termination and R mismatch
    #[test]
    fn dac_6581_nonlinear() {
        let table = build_dac_table(8, ChipModel::Mos6581);

        let mut is_linear = true;
        for i in 1..256 {
            if table[i] <= table[i - 1] {
                is_linear = false;
                break;
            }
        }

        assert!(!is_linear, "6581 DAC should be non-monotonic");
    }

    /// 8580 DAC is perfectly linear
    #[test]
    fn dac_8580_linear() {
        let table = build_dac_table(8, ChipModel::Mos8580);

        for i in 1..256 {
            assert!(
                table[i] > table[i - 1],
                "8580 DAC should be monotonic: table[{}]={} <= table[{}]={}",
                i,
                table[i],
                i - 1,
                table[i - 1]
            );
        }
    }

    /// MOSFET leakage causes non-zero output at input 0
    #[test]
    fn dac_leakage() {
        let table = build_dac_table(8, ChipModel::Mos6581);
        assert!(
            table[0] > 0.0,
            "Leakage should produce non-zero output at input 0"
        );
    }

    /// DAC output at max input should be close to 1.0 (normalized)
    #[test]
    fn dac_normalized() {
        let table_6581 = build_dac_table(8, ChipModel::Mos6581);
        let table_8580 = build_dac_table(8, ChipModel::Mos8580);

        assert!(
            (table_6581[255] - 1.0).abs() < 0.01,
            "6581 DAC max should be ~1.0: {}",
            table_6581[255]
        );
        assert!(
            (table_8580[255] - 1.0).abs() < 0.01,
            "8580 DAC max should be ~1.0: {}",
            table_8580[255]
        );
    }

    // =========================================================================
    // C++ libresidfp comparison tests
    // Reference values from libresidfp Dac::getOutput()
    // =========================================================================

    /// Asserts float is within tolerance of expected.
    macro_rules! assert_close {
        ($actual:expr, $expected:expr, $tol:expr, $msg:expr) => {{
            let diff = ($actual - $expected).abs();
            assert!(
                diff <= $tol,
                "{}: expected {:.6}, got {:.6} (diff: {:.6})",
                $msg,
                $expected,
                $actual,
                diff
            );
        }};
    }

    /// Compare 6581 8-bit DAC against C++ reference values.
    #[test]
    fn compare_cpp_dac_6581_8bit() {
        let table = build_dac_table(8, ChipModel::Mos6581);
        const TOL: f32 = 0.0001;

        // C++ reference: libresidfp Dac(8).kinkedDac(MOS6581).getOutput(i)
        assert_close!(table[0], 0.007500, TOL, "6581[0]");
        assert_close!(table[1], 0.014576, TOL, "6581[1]");
        assert_close!(table[2], 0.017792, TOL, "6581[2]");
        assert_close!(table[4], 0.025686, TOL, "6581[4]");
        assert_close!(table[8], 0.041846, TOL, "6581[8]");
        assert_close!(table[16], 0.073619, TOL, "6581[16]");
        assert_close!(table[32], 0.135445, TOL, "6581[32]");
        assert_close!(table[64], 0.255429, TOL, "6581[64]");
        assert_close!(table[128], 0.488107, TOL, "6581[128]");
        assert_close!(table[255], 1.000000, TOL, "6581[255]");
    }

    /// Compare 8580 8-bit DAC against C++ reference values.
    #[test]
    fn compare_cpp_dac_8580_8bit() {
        let table = build_dac_table(8, ChipModel::Mos8580);
        const TOL: f32 = 0.0001;

        // C++ reference: libresidfp Dac(8).kinkedDac(CSG8580).getOutput(i)
        assert_close!(table[0], 0.003500, TOL, "8580[0]");
        assert_close!(table[1], 0.007408, TOL, "8580[1]");
        assert_close!(table[2], 0.011316, TOL, "8580[2]");
        assert_close!(table[4], 0.019131, TOL, "8580[4]");
        assert_close!(table[8], 0.034763, TOL, "8580[8]");
        assert_close!(table[16], 0.066025, TOL, "8580[16]");
        assert_close!(table[32], 0.128551, TOL, "8580[32]");
        assert_close!(table[64], 0.253602, TOL, "8580[64]");
        assert_close!(table[128], 0.503704, TOL, "8580[128]");
        assert_close!(table[255], 1.000000, TOL, "8580[255]");
    }

    /// Compare 6581 11-bit DAC (filter cutoff) against C++ reference values.
    #[test]
    fn compare_cpp_dac_6581_11bit() {
        let table = build_dac_table(11, ChipModel::Mos6581);
        const TOL: f32 = 0.0001;

        // C++ reference: libresidfp Dac(11).kinkedDac(MOS6581).getOutput(i)
        assert_close!(table[0], 0.007500, TOL, "6581[0]");
        assert_close!(table[1], 0.008471, TOL, "6581[1]");
        assert_close!(table[4], 0.009996, TOL, "6581[4]");
        assert_close!(table[16], 0.016573, TOL, "6581[16]");
        assert_close!(table[64], 0.041521, TOL, "6581[64]");
        assert_close!(table[256], 0.135356, TOL, "6581[256]");
        assert_close!(table[512], 0.255378, TOL, "6581[512]");
        assert_close!(table[1024], 0.488073, TOL, "6581[1024]");
        assert_close!(table[2047], 1.000000, TOL, "6581[2047]");
    }

    /// Compare 8580 11-bit DAC (filter cutoff) against C++ reference values.
    #[test]
    fn compare_cpp_dac_8580_11bit() {
        let table = build_dac_table(11, ChipModel::Mos8580);
        const TOL: f32 = 0.0001;

        // C++ reference: libresidfp Dac(11).kinkedDac(CSG8580).getOutput(i)
        assert_close!(table[0], 0.003500, TOL, "8580[0]");
        assert_close!(table[1], 0.003987, TOL, "8580[1]");
        assert_close!(table[4], 0.005447, TOL, "8580[4]");
        assert_close!(table[16], 0.011289, TOL, "8580[16]");
        assert_close!(table[64], 0.034656, TOL, "8580[64]");
        assert_close!(table[256], 0.128123, TOL, "8580[256]");
        assert_close!(table[512], 0.252747, TOL, "8580[512]");
        assert_close!(table[1024], 0.501993, TOL, "8580[1024]");
        assert_close!(table[2047], 1.000000, TOL, "8580[2047]");
    }

    /// Verify specific non-monotonic points in 6581 8-bit DAC.
    /// These are the exact points where output decreases as input increases.
    #[test]
    fn compare_cpp_dac_6581_nonmonotonic_points() {
        let table = build_dac_table(8, ChipModel::Mos6581);
        const TOL: f32 = 0.0001;

        // C++ reference: specific non-monotonic transitions
        // At powers of 2, the output drops due to missing termination
        assert_close!(table[7], 0.043053, TOL, "6581[7]");
        assert_close!(table[8], 0.041846, TOL, "6581[8]");
        assert!(table[7] > table[8], "7->8 should decrease");

        assert_close!(table[15], 0.077399, TOL, "6581[15]");
        assert_close!(table[16], 0.073619, TOL, "6581[16]");
        assert!(table[15] > table[16], "15->16 should decrease");

        assert_close!(table[31], 0.143518, TOL, "6581[31]");
        assert_close!(table[32], 0.135445, TOL, "6581[32]");
        assert!(table[31] > table[32], "31->32 should decrease");

        assert_close!(table[63], 0.271464, TOL, "6581[63]");
        assert_close!(table[64], 0.255429, TOL, "6581[64]");
        assert!(table[63] > table[64], "63->64 should decrease");

        assert_close!(table[127], 0.519393, TOL, "6581[127]");
        assert_close!(table[128], 0.488107, TOL, "6581[128]");
        assert!(table[127] > table[128], "127->128 should decrease");
    }

    /// Verify 8580 DAC has uniform steps (linear behavior).
    #[test]
    fn compare_cpp_dac_8580_linear_steps() {
        let table = build_dac_table(8, ChipModel::Mos8580);

        // 8580 should have roughly uniform step sizes
        // Each power of 2 should be approximately 2x the previous
        let step_1 = table[1] - table[0];
        let step_2 = table[2] - table[1];
        let ratio = step_2 / step_1;

        // Steps should be nearly identical (ratio ~1.0)
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "8580 should have uniform steps, ratio: {}",
            ratio
        );

        // Power of 2 values should roughly double
        let ratio_2_1 = table[2] / table[1];
        let ratio_4_2 = table[4] / table[2];
        let ratio_8_4 = table[8] / table[4];

        assert!((ratio_2_1 - 1.5).abs() < 0.1, "2/1 ratio: {}", ratio_2_1);
        assert!((ratio_4_2 - 1.7).abs() < 0.1, "4/2 ratio: {}", ratio_4_2);
        assert!((ratio_8_4 - 1.8).abs() < 0.1, "8/4 ratio: {}", ratio_8_4);
    }
}
