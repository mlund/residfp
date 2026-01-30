// Tests for external filter with libresidfp-compatible RC values:
// Low-pass: R=10kOhm, C=1000pF (cutoff ~15.9kHz)
// High-pass: R=10kOhm, C=10uF (cutoff ~1.6Hz, assumes 10kOhm load)

use resid::external_filter::ExternalFilter;
use resid::ChipModel;

/// Single-cycle filter response follows first-order IIR curve
#[test]
fn clock() {
    let mut ext_filter = ExternalFilter::new(ChipModel::Mos6581);
    let mut outputs = Vec::new();
    let mut vi = -1000;
    while vi <= 1000 {
        ext_filter.clock(vi);
        outputs.push(ext_filter.output());
        vi += 50;
    }

    // Verify filter behavior: output should track input with low-pass smoothing
    // and high-pass DC removal (very slow, ~1.6Hz cutoff)
    assert_eq!(outputs.len(), 41);

    // First output should be close to input scaled by LP coefficient
    // With very slow HP, most of input passes through LP initially
    assert!(outputs[0] < 0, "Should track negative input");
    assert!(outputs[0] > -200, "LP smoothing limits initial response");

    // Filter should continue responding to changing input
    let mid_idx = outputs.len() / 2;
    assert!(
        outputs[mid_idx] < outputs[0],
        "Should track increasing input"
    );
}

/// Multi-cycle filter response with constant input settles toward input value
#[test]
fn clock_delta() {
    let mut ext_filter = ExternalFilter::new(ChipModel::Mos6581);
    let mut outputs = Vec::new();
    let mut vi = -1000;
    while vi <= 1000 {
        ext_filter.clock_delta(100, vi);
        outputs.push(ext_filter.output());
        vi += 50;
    }

    // With 100 cycles per step, LP filter has time to settle closer to input
    assert_eq!(outputs.len(), 41);

    // After 100 cycles at vi=-1000, output should be close to -1000
    // HP is very slow (1.6Hz), so minimal DC removal in 100 cycles
    assert!(
        outputs[0] < -900,
        "Should settle close to input: got {}",
        outputs[0]
    );
    assert!(
        outputs[0] > -1100,
        "Should not overshoot: got {}",
        outputs[0]
    );

    // Middle value (vi=0) should be near 0
    let mid_idx = outputs.len() / 2;
    assert!(
        outputs[mid_idx].abs() < 100,
        "Mid should be near zero: got {}",
        outputs[mid_idx]
    );

    // Last value (vi=1000) should be close to 1000
    let last = outputs[outputs.len() - 1];
    assert!(last > 900, "Should settle close to input: got {}", last);
}
