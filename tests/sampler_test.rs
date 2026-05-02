// Tests ported from libresidfp TestResampler.cpp
//
// Verify soft clipping behavior for 16-bit saturation.

use residfp::sampler::soft_clip;
use residfp::{ChipModel, SamplingMethod, Sid};

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

/// Extreme values clip near the i16 boundary with the correct sign.
#[test]
fn soft_clip_extremes() {
    // i16 bounds are guaranteed by the return type; assert the meaningful
    // invariant that extremes saturate near (not below) the soft-clip threshold.
    let max_clipped = soft_clip(i32::MAX);
    assert!(
        max_clipped as i32 >= THRESHOLD,
        "i32::MAX should saturate at >= threshold, got {}",
        max_clipped
    );

    let min_clipped = soft_clip(i32::MIN + 1);
    assert!(
        min_clipped as i32 <= -THRESHOLD,
        "i32::MIN+1 should saturate at <= -threshold, got {}",
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

/// `SamplingMethod::None` emits one output per SID cycle (raw 1 MHz feed).
#[test]
fn passthrough_one_sample_per_cycle() {
    let mut sid = Sid::new(ChipModel::Mos6581);
    sid.set_sampling_parameters(SamplingMethod::None, 1_000_000, 1_000_000)
        .unwrap();
    let mut buf = [0i16; 256];
    let (written, remaining) = sid.sample(256, &mut buf, 1);
    assert_eq!(written, 256);
    assert_eq!(remaining, 0);
}

/// `SamplingMethod::None` matches stepping the SID one cycle at a time.
#[test]
fn passthrough_matches_manual_clocking() {
    let mut reference = Sid::new(ChipModel::Mos6581);
    reference.write(0x05, 0x09); // attack/decay
    reference.write(0x18, 0x0f); // volume
    reference.write(0x01, 25); // freq hi
    reference.write(0x00, 177); // freq lo
    reference.write(0x04, 0x21); // sawtooth + gate
    let mut expected = [0i16; 128];
    for slot in expected.iter_mut() {
        reference.clock();
        *slot = reference.output();
    }

    let mut sid = Sid::new(ChipModel::Mos6581);
    sid.write(0x05, 0x09);
    sid.write(0x18, 0x0f);
    sid.write(0x01, 25);
    sid.write(0x00, 177);
    sid.write(0x04, 0x21);
    sid.set_sampling_parameters(SamplingMethod::None, 1_000_000, 1_000_000)
        .unwrap();
    let mut got = [0i16; 128];
    let (written, _) = sid.sample(128, &mut got, 1);
    assert_eq!(written, 128);
    assert_eq!(got, expected);
}

/// Buffer smaller than `delta` returns leftover cycles for the next call.
#[test]
fn passthrough_leftover_cycles() {
    let mut sid = Sid::new(ChipModel::Mos6581);
    sid.set_sampling_parameters(SamplingMethod::None, 1_000_000, 1_000_000)
        .unwrap();
    let mut buf = [0i16; 64];
    let (written, remaining) = sid.sample(100, &mut buf, 1);
    assert_eq!(written, 64);
    assert_eq!(remaining, 36);
}
