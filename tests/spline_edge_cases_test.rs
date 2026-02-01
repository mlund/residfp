// Tests ported from libresidfp TestSpline.cpp
//
// Verify spline interpolation for filter frequency curves.

use residfp::spline::{interpolate, Point, PointPlotter};

/// Hardware-measured opamp voltage response for 6581 filter.
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

/// Opamp curve must be monotonically decreasing (inverting amplifier behavior).
/// Non-monotonic splines would cause filter instability.
#[test]
fn monotonicity() {
    let points: Vec<Point> = OPAMP_VOLTAGE.iter().map(|&(x, y)| Point { x, y }).collect();

    // Evaluate spline at fine resolution
    let step = 0.01;
    let x_start = OPAMP_VOLTAGE[0].0;
    let max_x = OPAMP_VOLTAGE[OPAMP_VOLTAGE.len() - 1].0;

    // Create a dense output to check monotonicity
    let output_size = ((max_x - x_start) / step) as usize + 10;
    let mut output = vec![0i32; output_size];
    let mut plotter = PointPlotter::new(&mut output);
    interpolate(&points, &mut plotter, step);

    // Check monotonicity in the interpolated region
    // Note: We check from index 1 since index 0 might not be in the interpolated range
    let start_idx = (OPAMP_VOLTAGE[1].0 / step) as usize;
    let end_idx = (OPAMP_VOLTAGE[OPAMP_VOLTAGE.len() - 2].0 / step) as usize;

    let mut prev_y = output[start_idx] as f64;
    for i in (start_idx + 1)..end_idx.min(output.len()) {
        let y = output[i] as f64;
        // Allow equal values (plateau) but not increasing
        assert!(
            y <= prev_y || (y - prev_y).abs() < 0.001,
            "Spline should be monotonically decreasing: y[{}]={} > y[{}]={}",
            i,
            y,
            i - 1,
            prev_y
        );
        prev_y = y;
    }
}

/// Spline must pass through measured control points to match real hardware.
#[test]
fn control_points() {
    // Use integer x coordinates for this test (like the F0 curve)
    static TEST_POINTS: [(i32, i32); 9] = [
        (0, 100),
        (0, 100), // repeated start
        (10, 200),
        (20, 350),
        (30, 500),
        (40, 600),
        (50, 650),
        (50, 650), // repeated end
        (50, 650),
    ];

    let points: Vec<Point> = TEST_POINTS
        .iter()
        .map(|&(x, y)| Point {
            x: x as f64,
            y: y as f64,
        })
        .collect();

    let mut output = vec![0i32; 60];
    let mut plotter = PointPlotter::new(&mut output);
    interpolate(&points, &mut plotter, 1.0);

    // Check each integer control point (skip the repeated boundary points)
    for &(x, expected_y) in &TEST_POINTS[2..6] {
        let idx = x as usize;
        if idx < output.len() {
            let actual_y = output[idx];
            let tolerance = 2; // Allow small interpolation errors
            assert!(
                (actual_y - expected_y).abs() <= tolerance,
                "Point ({}, {}) not on spline: actual y = {}",
                x,
                expected_y,
                actual_y
            );
        }
    }
}
