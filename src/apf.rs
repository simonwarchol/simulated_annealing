//! Provides the [`APF`](crate::APF) enum

use num::Float;
use numeric_literals::replace_float_literals;
use rand::prelude::*;
use rand_distr::{uniform::SampleUniform, Uniform};

use std::fmt::Debug;

/// Acceptance probability function
pub enum APF<F, R>
where
    F: Float + SampleUniform,
    R: Rng,
{
    /// Metropolis criterion:
    ///
    /// $
    /// P(\Delta f, t) = \begin{cases}
    /// 1, & if \\; \Delta f \leqslant 0; \\\\
    /// \min(e^{- \Delta f / t}, 1), & if \\; \Delta f \gt 0
    /// \end{cases}
    /// $
    Metropolis,
    /// Custom: choose your own!
    Custom {
        /// Custom function
        f: fn(diff: F, t: F, uni: &Uniform<F>, rng: &mut R) -> bool,
    },
}

impl<F, R> APF<F, R>
where
    F: Float + SampleUniform + Debug,
    R: Rng,
{
    /// Choose whether to accept the point
    ///
    /// Arguments:
    /// * `diff` --- Difference in the objective;
    /// * `t` --- Temperature;
    /// * `uni` -- Uniform[0, 1] distribution;
    /// * `rng` --- Random number generator.
    #[replace_float_literals(F::from(literal).unwrap())]
    pub fn accept(&self, diff: F, t: F, uni: &Uniform<F>, rng: &mut R) -> bool {
        match self {
            APF::Metropolis => diff <= 0. || uni.sample(rng) < F::min(F::exp(-diff / t), 1.),
            APF::Custom { f } => f(diff, t, uni, rng),
        }
    }
}
