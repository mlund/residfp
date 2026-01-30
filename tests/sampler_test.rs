// Tests ported from libresidfp TestResampler.cpp
//
// Verify soft clipping behavior for 16-bit saturation.

use residfp::sampler::soft_clip;

/// Soft clipping threshold - values below pass unchanged.
const THRESHOLD: i32 = 28000;

/// Peak value for testing compression region.
const PEAK: i32 = 38000;

/// Values within threshold pass unchanged (linear region).
#[test]
fn soft_clip_linear_region() {
    for i in -THRESHOLD..=THRESHOLD {
        let clipped = soft_clip(i);
        assert_eq!(
            clipped, i as i16,
            "Value {} in linear region should pass unchanged, got {}",
            i, clipped
        );
    }
}

/// Positive values above threshold are compressed but stay <= i16::MAX.
#[test]
fn soft_clip_positive_compression() {
    for i in THRESHOLD..=PEAK {
        let clipped = soft_clip(i) as i32;
        assert!(
            clipped <= i && clipped <= i16::MAX as i32,
            "Positive {} should compress: got {}",
            i,
            clipped
        );
    }
}

/// Negative values below -threshold are compressed but stay >= i16::MIN.
#[test]
fn soft_clip_negative_compression() {
    for i in (-PEAK..=-THRESHOLD).rev() {
        let clipped = soft_clip(i) as i32;
        assert!(
            clipped >= i && clipped >= i16::MIN as i32,
            "Negative {} should compress: got {}",
            i,
            clipped
        );
    }
}

/// Extreme values stay within i16 range.
#[test]
fn soft_clip_extremes() {
    let max_clipped = soft_clip(i32::MAX);
    assert!(
        max_clipped <= i16::MAX,
        "i32::MAX should clip to <= i16::MAX, got {}",
        max_clipped
    );

    let min_clipped = soft_clip(i32::MIN + 1);
    assert!(
        min_clipped >= i16::MIN,
        "i32::MIN+1 should clip to >= i16::MIN, got {}",
        min_clipped
    );
}

/// Soft clipping is monotonic (larger input -> larger or equal output).
#[test]
fn soft_clip_monotonic() {
    let mut prev = soft_clip(-100000);
    for i in -100000..=100000 {
        let curr = soft_clip(i);
        assert!(
            curr >= prev,
            "Soft clip should be monotonic: f({}) = {} < f({}) = {}",
            i - 1,
            prev,
            i,
            curr
        );
        prev = curr;
    }
}

/// Symmetry: soft_clip(-x) approximately equals -soft_clip(x).
/// Not exact due to asymmetric i16 range (-32768 vs 32767) and different max_val.
#[test]
fn soft_clip_symmetry() {
    for i in 0..=THRESHOLD {
        // Linear region should be exactly symmetric
        let pos = soft_clip(i);
        let neg = soft_clip(-i);
        assert_eq!(
            pos as i32,
            -(neg as i32),
            "Linear region should be symmetric: f({}) = {}, f({}) = {}",
            i,
            pos,
            -i,
            neg
        );
    }
    // Compression region: verify both sides compress similarly (not exact)
    for i in (THRESHOLD + 1000)..PEAK {
        let pos = soft_clip(i);
        let neg = soft_clip(-i);
        // Both should be compressed (output magnitude less than input)
        assert!(
            (pos as i32) < i,
            "Positive {} should compress to less than input, got {}",
            i,
            pos
        );
        assert!(
            (neg as i32) > -i,
            "Negative {} should compress to less than input magnitude, got {}",
            -i,
            neg
        );
    }
}
