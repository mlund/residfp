// Tests for TwoPassResampler.
//
// Validates that the two-pass sinc resampler produces correct output
// and matches the behavior of the single-pass resampler.

use residfp::sampler::SamplingMethod;
use residfp::{ChipModel, Sid};

/// PAL C64 clock frequency
const CLOCK_FREQ: u32 = 985248;

/// Output sample rate
const SAMPLE_FREQ: u32 = 48000;

/// Create SID with two-pass resampling at given frequencies.
fn create_two_pass_sid(clock_freq: u32, sample_freq: u32) -> Sid {
    let mut sid = Sid::new(ChipModel::Mos6581);
    sid.set_sampling_parameters(SamplingMethod::ResampleTwoPass, clock_freq, sample_freq)
        .expect("Failed to set sampling parameters");
    sid
}

/// Configure SID voice 1 with a simple test tone (sawtooth, fast envelope).
fn setup_test_tone(sid: &mut Sid) {
    sid.write(0x18, 0x0f); // Volume = 15
    sid.write(0x05, 0x00); // Attack=0, Decay=0
    sid.write(0x06, 0xf0); // Sustain=15, Release=0
    sid.write(0x00, 0x00); // Freq lo
    sid.write(0x01, 0x20); // Freq hi
    sid.write(0x04, 0x21); // Gate on, sawtooth
}

/// Assert resampler works for given clock/sample rate combination.
macro_rules! assert_resampler_works {
    ($clock:expr, $sample:expr, $msg:expr) => {{
        let mut sid = Sid::new(ChipModel::Mos6581);
        let result = sid.set_sampling_parameters(SamplingMethod::ResampleTwoPass, $clock, $sample);
        assert!(result.is_ok(), "{}: init failed", $msg);

        sid.write(0x04, 0x11);
        let mut buffer = [0i16; 256];
        let (n, _) = sid.sample(1000, &mut buffer, 1);
        assert!(n > 0, "{}: no output", $msg);
    }};
}

/// Test that the two-pass resampler initializes without panic.
#[test]
fn two_pass_resampler_initializes() {
    let mut sid = Sid::new(ChipModel::Mos6581);
    let result =
        sid.set_sampling_parameters(SamplingMethod::ResampleTwoPass, CLOCK_FREQ, SAMPLE_FREQ);
    assert!(result.is_ok(), "Two-pass resampler should initialize");
}

/// Test that the two-pass resampler produces output.
#[test]
fn two_pass_resampler_produces_output() {
    let mut sid = create_two_pass_sid(CLOCK_FREQ, SAMPLE_FREQ);
    setup_test_tone(&mut sid);

    let mut buffer = [0i16; 8192];
    let mut total_samples = 0;
    let mut non_zero_samples = 0;

    // Clock many cycles to fill two-pass ring buffers (2048 samples each stage)
    for _ in 0..2000 {
        let (n, _) = sid.sample(CLOCK_FREQ / SAMPLE_FREQ, &mut buffer, 1);
        total_samples += n;
        non_zero_samples += buffer[..n].iter().filter(|&&s| s != 0).count();
    }

    assert!(total_samples > 0, "Should produce samples");
    assert!(non_zero_samples > 0, "Should produce non-zero output");
}

/// Test that two-pass resampler produces similar output to single-pass.
#[test]
fn two_pass_matches_single_pass_approximately() {
    fn generate_samples(method: SamplingMethod) -> Vec<i16> {
        let mut sid = Sid::new(ChipModel::Mos6581);
        sid.set_sampling_parameters(method, CLOCK_FREQ, SAMPLE_FREQ)
            .expect("Failed to set sampling parameters");

        // Enable a voice with a test tone (1kHz-ish)
        sid.write(0x05, 0x00); // Attack=0, Decay=0
        sid.write(0x06, 0xf0); // Sustain=15, Release=0
        sid.write(0x00, 0x00); // Freq lo
        sid.write(0x01, 0x08); // Freq hi
        sid.write(0x04, 0x11); // Gate on, triangle wave

        let mut all_samples = Vec::new();
        let mut buffer = [0i16; 1024];

        // Generate 1000 output samples worth of cycles
        let cycles_per_sample = CLOCK_FREQ / SAMPLE_FREQ;
        for _ in 0..1000 {
            let (n, _) = sid.sample(cycles_per_sample, &mut buffer, 1);
            all_samples.extend_from_slice(&buffer[..n]);
        }

        all_samples
    }

    let single_pass = generate_samples(SamplingMethod::Resample);
    let two_pass = generate_samples(SamplingMethod::ResampleTwoPass);

    // Both should produce roughly the same number of samples
    let len_diff = (single_pass.len() as i32 - two_pass.len() as i32).abs();
    assert!(
        len_diff < 10,
        "Sample count should be similar: single={}, two_pass={}",
        single_pass.len(),
        two_pass.len()
    );

    // Compare RMS power - should be within 3dB
    let single_power: f64 = single_pass.iter().map(|&s| (s as f64).powi(2)).sum();
    let two_power: f64 = two_pass.iter().map(|&s| (s as f64).powi(2)).sum();

    if single_power > 0.0 && two_power > 0.0 {
        let power_ratio_db = 10.0 * (two_power / single_power).log10();
        assert!(
            power_ratio_db.abs() < 3.0,
            "Power difference should be within 3dB, got {:.2}dB",
            power_ratio_db
        );
    }
}

/// Test reset clears resampler state.
#[test]
fn two_pass_reset_clears_state() {
    let mut sid = create_two_pass_sid(CLOCK_FREQ, SAMPLE_FREQ);
    let mut buffer = [0i16; 1024];

    // Generate samples then reset
    sid.write(0x04, 0x11);
    for _ in 0..50 {
        let _ = sid.sample(100, &mut buffer, 1);
    }
    sid.reset();

    // After reset, output should be silent (no clicks from previous state)
    let (n, _) = sid.sample(100, &mut buffer, 1);
    for (i, &sample) in buffer[..n].iter().enumerate() {
        assert!(
            sample.abs() < 100,
            "Sample {} after reset should be near zero, got {}",
            i,
            sample
        );
    }
}

/// Test various sample rates work correctly.
#[test]
fn two_pass_various_sample_rates() {
    assert_resampler_works!(CLOCK_FREQ, 22050, "22050 Hz");
    assert_resampler_works!(CLOCK_FREQ, 44100, "44100 Hz");
    assert_resampler_works!(CLOCK_FREQ, 48000, "48000 Hz");
    assert_resampler_works!(CLOCK_FREQ, 96000, "96000 Hz");
}

/// Test NTSC clock frequency.
#[test]
fn two_pass_ntsc_clock() {
    const NTSC_CLOCK: u32 = 1022730;
    assert_resampler_works!(NTSC_CLOCK, SAMPLE_FREQ, "NTSC clock");
}
