//! ODE schedule and the seeded noise sampler for LuxTTS flow matching.
//!
//! `time_steps` is the numpy port of the reference `get_time_steps`; the
//! schedule transform `t_shift * t / (1 + (t_shift - 1) * t)` concentrates
//! integration steps near t=1, where the distilled model is most sensitive.
//!
//! [`StandardNormal`] is a PCG64 + Box-Muller host-side sampler, matching the
//! approach the Inflect-Micro-v2 frontend takes: any correctly-distributed
//! noise yields valid audio, but a given seed here does not select the same
//! sample as the same seed in the NumPy reference (only NumPy's exact ziggurat
//! stream would reproduce that, and reproducing it buys nothing - the seed
//! only picks which sample from the distribution you get).

/// Shifted uniform schedule: `num_steps + 1` points from t_start to t_end.
pub fn time_steps(num_steps: usize, t_shift: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = i as f64 / num_steps.max(1) as f64;
        let shifted = t_shift as f64 * t / (1.0 + (t_shift as f64 - 1.0) * t);
        out.push(shifted as f32);
    }
    out
}

/// Seeded standard-normal sampler (PCG64 output function, Box-Muller transform).
pub struct StandardNormal {
    state: u128,
    spare: Option<f32>,
}

impl StandardNormal {
    const MULTIPLIER: u128 = 47026247687942121848144207491837523525;
    const INCREMENT: u128 = 117397592171526113268558934119004209487;

    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: 0, spare: None };
        rng.state = rng
            .state
            .wrapping_add(Self::INCREMENT)
            .wrapping_add(seed as u128);
        rng.step();
        rng
    }

    fn step(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        let xored = ((self.state >> 64) ^ self.state) as u64;
        let rot = (self.state >> 122) as u32;
        xored.rotate_right(rot)
    }

    fn next_open_unit(&mut self) -> f64 {
        let bits = self.step() >> 11;
        (bits as f64 + 1.0) / (9007199254740992.0 + 1.0)
    }

    pub fn next(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u1 = self.next_open_unit();
        let u2 = self.next_open_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some((radius * theta.sin()) as f32);
        (radius * theta.cos()) as f32
    }

    pub fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_steps_endpoints_and_monotonicity() {
        let ts = time_steps(8, 0.9);
        assert_eq!(ts.len(), 9);
        assert!(ts[0] >= 0.0 && ts[0] < 1e-6);
        assert!((ts[8] - 1.0).abs() < 1e-6);
        for w in ts.windows(2) {
            assert!(w[1] > w[0], "schedule must be strictly increasing");
        }
    }

    #[test]
    fn test_time_steps_shift_pulls_midpoints_down() {
        // t_shift < 1 compresses the schedule toward t=0 (the reference
        // transform 0.9*t/(1+(0.9-1)*t) is below the identity for 0<t<1).
        let shifted = time_steps(8, 0.9);
        let plain = time_steps(8, 1.0);
        for i in 1..8 {
            assert!(shifted[i] < plain[i], "t_shift<1 moves midpoint {i} down");
        }
    }

    #[test]
    fn test_standard_normal_is_deterministic_for_a_seed() {
        let a = StandardNormal::new(42).fill(64);
        let b = StandardNormal::new(42).fill(64);
        assert_eq!(a, b);
    }

    #[test]
    fn test_standard_normal_is_roughly_unit_gaussian() {
        let samples = StandardNormal::new(7).fill(20_000);
        let mean: f64 = samples.iter().map(|s| *s as f64).sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter().map(|s| (*s as f64 - mean).powi(2)).sum::<f64>()
            / samples.len() as f64;
        assert!(mean.abs() < 0.05);
        assert!((variance - 1.0).abs() < 0.1);
    }
}
