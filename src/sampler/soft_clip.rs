// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! Soft clipping for smooth audio saturation.
//!
//! Uses tanh approximation to create smooth saturation instead of hard clipping
//! when audio samples exceed the 16-bit range.

/// Soft clipping threshold - values below pass unchanged
const THRESHOLD: i32 = 28000;

/// Padé tanh approximation (5th order), accurate for |x| < 3.
#[inline]
fn tanh_pade(x: f64) -> f64 {
    if x.abs() < 3.0 {
        let x2 = x * x;
        let num = x * (945.0 + x2 * (105.0 + x2));
        let den = 945.0 + x2 * (420.0 + x2 * 15.0);
        num / den
    } else if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Soft clip positive value using tanh curve above threshold.
/// Creates smooth saturation instead of hard clipping.
#[inline]
fn soft_clip_positive(x: i32, max_val: i32) -> i32 {
    if x < THRESHOLD {
        return x;
    }

    let max_f = max_val as f64;
    let t = THRESHOLD as f64 / max_f;
    let a = 1.0 - t;
    let b = 1.0 / a;

    let value = (x - THRESHOLD) as f64 / max_f;
    let result = t + a * tanh_pade(b * value);
    (result * max_f) as i32
}

/// Soft clip into 16-bit range [i16::MIN, i16::MAX].
/// Uses tanh curve for smooth saturation above threshold.
#[inline]
pub fn soft_clip(x: i32) -> i16 {
    if x < 0 {
        // Handle i32::MIN overflow case
        let abs_x = if x == i32::MIN { i32::MAX } else { -x };
        // Negate as i32 first to avoid i16 overflow
        // Note: i16::MIN magnitude is 32768, one more than i16::MAX
        (-soft_clip_positive(abs_x, (i16::MIN as i32).abs())) as i16
    } else {
        soft_clip_positive(x, i16::MAX as i32) as i16
    }
}
