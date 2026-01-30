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
            ChipModel::Mos6581 => 0.0075,
            ChipModel::Mos8580 => 0.0035,
        }
    }

    /// 6581 has imperfect resistor matching, 8580 is ideal
    const fn r2r_ratio(self) -> f64 {
        match self {
            ChipModel::Mos6581 => 2.20,
            ChipModel::Mos8580 => 2.00,
        }
    }

    /// 6581 DACs lack termination resistor at bit 0
    const fn has_termination(self) -> bool {
        matches!(self, ChipModel::Mos8580)
    }
}

/// Parallel resistance: r1 || r2
fn parallel(r1: f64, r2: f64) -> f64 {
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
}
