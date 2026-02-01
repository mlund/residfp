// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! Two-pass sinc resampler using chained FIR filters.
//!
//! Chains two FIR filters through an intermediate frequency for improved
//! computational efficiency. The intermediate frequency is calculated using
//! Laurent Ganier's formula to minimize total filter order.

use alloc::vec::Vec;

use super::fir::{i0, sqrt_compat};

/// Ring buffer size for two-pass resampler.
/// Smaller than single-pass (2048 vs 16384) since each pass handles less decimation.
const RING_SIZE: usize = 2048;
const RING_MASK: usize = RING_SIZE - 1;

/// Fixed-point scale for two-pass resampler.
/// Uses 10 bits (vs 16 for single-pass) to match libresidfp's precision tradeoffs.
const FIXP_SHIFT: i32 = 10;
const FIXP_SCALE: i32 = 1 << FIXP_SHIFT;
const FIXP_MASK: i32 = FIXP_SCALE - 1;

/// Passband limit for high sample rates (>44kHz).
/// 20kHz covers full audible range while leaving transition band headroom.
const DEFAULT_PASSBAND: f64 = 20000.0;

/// FIR filter coefficients for two-pass resampling.
#[derive(Clone)]
struct Fir {
    data: Vec<i16>,
    n: i32,
    res: i32,
}

/// Two-pass sinc resampler using chained FIR filters.
#[derive(Clone)]
pub struct TwoPassResampler {
    /// Pass 1 FIR: clock_freq → intermediate_freq
    fir1: Fir,
    /// Pass 2 FIR: intermediate_freq → sample_freq
    fir2: Fir,
    /// Ring buffer for pass 1 (stores i32 for intermediate precision)
    buffer1: [i32; RING_SIZE * 2],
    /// Ring buffer for pass 2
    buffer2: [i32; RING_SIZE * 2],
    /// Current index in pass 1 ring buffer
    index1: usize,
    /// Current index in pass 2 ring buffer
    index2: usize,
    /// Sample offset for pass 1 (fixed-point)
    offset1: i32,
    /// Sample offset for pass 2 (fixed-point)
    offset2: i32,
    /// Cycles per sample for pass 1 (fixed-point)
    cycles_per_sample1: i32,
    /// Cycles per sample for pass 2 (fixed-point)
    cycles_per_sample2: i32,
    /// Output value from pass 1
    output_value1: i32,
    /// Final output value from pass 2
    output_value2: i32,
}

impl TwoPassResampler {
    /// Create a new two-pass resampler.
    pub fn new(clock_freq: f64, sample_freq: f64) -> Self {
        let passband = Self::passband_freq(sample_freq);
        let intermediate_freq = Self::calculate_intermediate_freq(clock_freq, sample_freq);

        let fir1 = Self::init_fir_pass(clock_freq, intermediate_freq, passband);
        let fir2 = Self::init_fir_pass(intermediate_freq, sample_freq, passband);

        let cycles_per_sample1 = (clock_freq / intermediate_freq * FIXP_SCALE as f64) as i32;
        let cycles_per_sample2 = (intermediate_freq / sample_freq * FIXP_SCALE as f64) as i32;

        Self {
            fir1,
            fir2,
            buffer1: [0; RING_SIZE * 2],
            buffer2: [0; RING_SIZE * 2],
            index1: 0,
            index2: 0,
            offset1: 0,
            offset2: 0,
            cycles_per_sample1,
            cycles_per_sample2,
            output_value1: 0,
            output_value2: 0,
        }
    }

    /// Process sample through both passes, return true if final output ready.
    #[inline]
    pub fn input(&mut self, sample: i32) -> bool {
        // Pass 1: clock_freq → intermediate_freq
        Self::store_in_ring(&mut self.buffer1, &mut self.index1, sample);

        if let Some(offset) = Self::check_decimation(&mut self.offset1, self.cycles_per_sample1) {
            self.output_value1 =
                self.convolve_with_fir(&self.fir1, &self.buffer1, self.index1, offset);

            // Pass 2: intermediate_freq → sample_freq (only when pass 1 outputs)
            Self::store_in_ring(&mut self.buffer2, &mut self.index2, self.output_value1);

            if let Some(offset2) =
                Self::check_decimation(&mut self.offset2, self.cycles_per_sample2)
            {
                self.output_value2 =
                    self.convolve_with_fir(&self.fir2, &self.buffer2, self.index2, offset2);
                return true;
            }
        }
        false
    }

    /// Get the final output value.
    #[inline]
    pub const fn output(&self) -> i32 {
        self.output_value2
    }

    /// Reset resampler state.
    pub const fn reset(&mut self) {
        self.buffer1 = [0; RING_SIZE * 2];
        self.buffer2 = [0; RING_SIZE * 2];
        self.index1 = 0;
        self.index2 = 0;
        self.offset1 = 0;
        self.offset2 = 0;
        self.output_value1 = 0;
        self.output_value2 = 0;
    }

    /// Determine passband frequency based on output sample rate.
    /// Uses 20kHz for high rates (full audible range), 45% of rate for low rates.
    fn passband_freq(sample_freq: f64) -> f64 {
        if sample_freq > 44000.0 {
            DEFAULT_PASSBAND
        } else {
            sample_freq * 0.45
        }
    }

    /// Calculate optimal intermediate frequency using Laurent Ganier's formula.
    /// Minimizes total filter order by balancing decimation between two stages.
    fn calculate_intermediate_freq(clock_freq: f64, sample_freq: f64) -> f64 {
        let passband = Self::passband_freq(sample_freq);
        // Formula derived from minimizing sum of filter orders for two-stage resampling
        2.0 * passband
            + (2.0 * passband * clock_freq * (sample_freq - 2.0 * passband) / sample_freq).sqrt()
    }

    /// Initialize FIR filter for one resampling pass.
    fn init_fir_pass(input_freq: f64, output_freq: f64, passband_freq: f64) -> Fir {
        let cycles_per_sample = input_freq / output_freq;
        let samples_per_cycle = output_freq / input_freq;

        // 16-bit target requires -96dB stopband attenuation
        let attenuation_db = -20.0_f64 * (1.0 / (1_i32 << 16) as f64).log10();

        // Transition band width (doubled since filter transitions at Nyquist)
        let transition_width =
            (1.0 - 2.0 * passband_freq / output_freq) * core::f64::consts::PI * 2.0;

        // Kaiser window parameters from MATLAB kaiserord function
        let beta = 0.1102_f64 * (attenuation_db - 8.7);
        let i0_beta = i0(beta);

        // Filter order: even for symmetric sinc around x=0
        let mut filter_order = ((attenuation_db - 7.95) / (2.285 * transition_width) + 0.5) as i32;
        filter_order += filter_order & 1;

        // Filter length: odd for symmetric sinc (order + 1)
        let mut filter_len = (filter_order as f64 * cycles_per_sample) as i32 + 1;
        filter_len |= 1;

        // Resolution bounded by interpolation error formula: err < 1.234/L^2
        let filter_res = ((1.234_f64 * (1 << 16) as f64).sqrt() * samples_per_cycle).ceil() as i32;

        let data = Self::compute_fir_coefficients(
            filter_len,
            filter_res,
            samples_per_cycle,
            beta,
            i0_beta,
        );

        Fir {
            data,
            n: filter_len,
            res: filter_res,
        }
    }

    /// Compute Kaiser-windowed sinc FIR coefficients.
    fn compute_fir_coefficients(
        filter_len: i32,
        filter_res: i32,
        samples_per_cycle: f64,
        beta: f64,
        i0_beta: f64,
    ) -> Vec<i16> {
        let mut data = alloc::vec![0i16; (filter_len * filter_res) as usize];
        let half_len = filter_len / 2;
        let half_len_f = half_len as f64;
        // Scale factor: 32768 for i16 range, samples_per_cycle for gain normalization
        let scale = 32768.0 * samples_per_cycle;

        for phase_idx in 0..filter_res {
            let phase_offset = phase_idx as f64 / filter_res as f64 + half_len_f;

            for tap_idx in 0..filter_len {
                let tap_offset = tap_idx as f64 - phase_offset;
                let normalized_pos = tap_offset / half_len_f;

                let kaiser_weight = Self::kaiser_window(normalized_pos, beta, i0_beta);
                let sinc_value = Self::sinc(tap_offset * samples_per_cycle * core::f64::consts::PI);

                data[(phase_idx * filter_len + tap_idx) as usize] =
                    (scale * sinc_value * kaiser_weight) as i16;
            }
        }
        data
    }

    /// Kaiser window function for the given normalized position [-1, 1].
    #[inline]
    fn kaiser_window(normalized_pos: f64, beta: f64, i0_beta: f64) -> f64 {
        if normalized_pos.abs() < 1.0 {
            i0(beta * sqrt_compat(1.0 - normalized_pos * normalized_pos)) / i0_beta
        } else {
            0.0
        }
    }

    /// Normalized sinc function: sin(x)/x, with sinc(0) = 1.
    #[inline]
    fn sinc(x: f64) -> f64 {
        if x.abs() >= 1e-8 {
            x.sin() / x
        } else {
            1.0
        }
    }

    /// Store sample in ring buffer with duplication for wraparound optimization.
    #[inline]
    fn store_in_ring(buffer: &mut [i32], index: &mut usize, sample: i32) {
        buffer[*index] = sample;
        buffer[*index + RING_SIZE] = sample;
        *index = (*index + 1) & RING_MASK;
    }

    /// Check if decimation produces output, update offset state.
    /// Returns offset for convolution when output is ready.
    #[inline]
    const fn check_decimation(offset: &mut i32, cycles_per_sample: i32) -> Option<i32> {
        let convolution_offset = if *offset < FIXP_SCALE {
            let result = *offset;
            *offset += cycles_per_sample;
            Some(result)
        } else {
            None
        };
        *offset -= FIXP_SCALE;
        convolution_offset
    }

    /// FIR convolution with linear interpolation between adjacent filter tables.
    fn convolve_with_fir(&self, fir: &Fir, buffer: &[i32], index: usize, subcycle: i32) -> i32 {
        // Select two adjacent FIR tables for interpolation
        let mut table_idx = (subcycle * fir.res) >> FIXP_SHIFT;
        let interp_frac = (subcycle * fir.res) & FIXP_MASK;

        // Ring buffer index for most recent samples (duplicated region avoids wrap check)
        let mut sample_idx = index + RING_SIZE - fir.n as usize;

        let v1 = Self::dot_product_i32(
            &buffer[sample_idx..sample_idx + fir.n as usize],
            &fir.data[(table_idx * fir.n) as usize..],
            fir.n as usize,
        );

        // Next table; wrap to first table requires shifting sample window
        table_idx += 1;
        if table_idx == fir.res {
            table_idx = 0;
            sample_idx += 1;
        }

        let v2 = Self::dot_product_i32(
            &buffer[sample_idx..sample_idx + fir.n as usize],
            &fir.data[(table_idx * fir.n) as usize..],
            fir.n as usize,
        );

        // Linear interpolation: v1 + frac * (v2 - v1)
        v1 + ((interp_frac * (v2 - v1)) >> FIXP_SHIFT)
    }

    /// Dot product of i32 samples with i16 coefficients.
    #[inline]
    fn dot_product_i32(samples: &[i32], coeffs: &[i16], len: usize) -> i32 {
        let mut acc: i64 = 0;
        for i in 0..len {
            acc += samples[i] as i64 * coeffs[i] as i64;
        }
        // Rounding shift: add half the divisor before shifting
        ((acc + (1 << 14)) >> 15) as i32
    }
}
