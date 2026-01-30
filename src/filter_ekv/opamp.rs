// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! Fritsch-Carlson monotone cubic spline interpolation.
//!
//! Used to create the op-amp reverse transfer function lookup table for the EKV
//! filter model. Based on the algorithm from Wikipedia's "Monotone cubic interpolation".

use alloc::vec::Vec;

/// A 2D point for spline interpolation.
#[derive(Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Cubic polynomial coefficients for one spline segment.
struct SplineSegment {
    x1: f64,
    x2: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

/// Monotone cubic spline interpolator using Fritsch-Carlson method.
pub struct MonotoneSpline {
    segments: Vec<SplineSegment>,
}

impl MonotoneSpline {
    /// Constructs a monotone cubic spline from input points.
    ///
    /// Points must be sorted by x coordinate in ascending order.
    pub fn new(input: &[Point]) -> Self {
        assert!(input.len() > 2, "Spline requires at least 3 points");

        let n = input.len() - 1;
        let mut segments = Vec::with_capacity(n);

        // Compute consecutive differences and slopes
        let dxs: Vec<f64> = (0..n).map(|i| input[i + 1].x - input[i].x).collect();
        let ms: Vec<f64> = (0..n)
            .map(|i| (input[i + 1].y - input[i].y) / dxs[i])
            .collect();

        // Compute degree-1 coefficients (tangents at each point)
        let mut cs = Vec::with_capacity(n + 1);
        cs.push(ms[0]);
        for i in 1..n {
            let m = ms[i - 1];
            let m_next = ms[i];
            if m * m_next <= 0.0 {
                // Sign change or zero slope: set tangent to zero for monotonicity
                cs.push(0.0);
            } else {
                // Weighted harmonic mean of slopes
                let dx = dxs[i - 1];
                let dx_next = dxs[i];
                let common = dx + dx_next;
                cs.push(3.0 * common / ((common + dx_next) / m + (common + dx) / m_next));
            }
        }
        cs.push(ms[n - 1]);

        // Build spline segments with polynomial coefficients
        for i in 0..n {
            let c1 = cs[i];
            let m = ms[i];
            let inv_dx = 1.0 / dxs[i];
            let common = c1 + cs[i + 1] - m - m;

            segments.push(SplineSegment {
                x1: input[i].x,
                x2: if i == n - 1 { f64::MAX } else { input[i + 1].x },
                d: input[i].y,
                c: c1,
                b: (m - c1 - common) * inv_dx,
                a: common * inv_dx * inv_dx,
            });
        }

        MonotoneSpline { segments }
    }

    /// Evaluates the spline at x, returning (y, dy/dx).
    pub fn evaluate(&self, x: f64) -> (f64, f64) {
        // Find the appropriate segment
        let seg = self
            .segments
            .iter()
            .find(|s| x <= s.x2)
            .unwrap_or(self.segments.last().unwrap());

        let diff = x - seg.x1;

        // y = a*x^3 + b*x^2 + c*x + d (in terms of diff from x1)
        let y = ((seg.a * diff + seg.b) * diff + seg.c) * diff + seg.d;

        // dy/dx = 3*a*x^2 + 2*b*x + c
        let dy = (3.0 * seg.a * diff + 2.0 * seg.b) * diff + seg.c;

        (y, dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Op-amp voltage data from MOS 6581R4AR (same as libresidfp TestSpline.cpp)
    const OPAMP_VOLTAGE: [(f64, f64); 33] = [
        (0.81, 10.31),
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
        (4.54, 4.54),
        (4.60, 4.19),
        (4.80, 3.00),
        (4.90, 2.30),
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
        (10.31, 0.81),
    ];

    fn opamp_points() -> Vec<Point> {
        OPAMP_VOLTAGE.iter().map(|&(x, y)| Point { x, y }).collect()
    }

    /// Mimics libresidfp TestSpline.cpp TestMonotonicity
    /// Op-amp transfer function is monotonically decreasing
    #[test]
    fn opamp_monotonicity() {
        let spline = MonotoneSpline::new(&opamp_points());

        let mut prev_y = f64::MAX;
        let mut x = 0.0;
        while x < 12.0 {
            let (y, _) = spline.evaluate(x);
            assert!(
                y <= prev_y,
                "Spline should be monotone decreasing at x={}: {} > {}",
                x,
                y,
                prev_y
            );
            prev_y = y;
            x += 0.01;
        }
    }

    /// Mimics libresidfp TestSpline.cpp TestPoints
    /// Spline must pass through all input points exactly
    #[test]
    fn opamp_interpolates_all_points() {
        let points = opamp_points();
        let spline = MonotoneSpline::new(&points);

        for (i, point) in points.iter().enumerate() {
            let (y, _) = spline.evaluate(point.x);
            assert!(
                (y - point.y).abs() < 1e-10,
                "Point {} at x={}: expected {}, got {}",
                i,
                point.x,
                point.y,
                y
            );
        }
    }

    /// Mimics libresidfp TestSpline.cpp TestInterpolateOutsideBounds
    #[test]
    fn interpolate_outside_bounds() {
        let points = [
            Point { x: 10.0, y: 15.0 },
            Point { x: 15.0, y: 20.0 },
            Point { x: 20.0, y: 30.0 },
            Point { x: 25.0, y: 40.0 },
            Point { x: 30.0, y: 45.0 },
        ];
        let spline = MonotoneSpline::new(&points);

        // Extrapolate below range
        let (y, _) = spline.evaluate(5.0);
        assert!(
            (y - 6.66667).abs() < 0.00001,
            "Below range: expected 6.66667, got {}",
            y
        );

        // Extrapolate above range
        let (y, _) = spline.evaluate(40.0);
        assert!(
            (y - 75.0).abs() < 0.00001,
            "Above range: expected 75.0, got {}",
            y
        );
    }
}
