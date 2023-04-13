//! Provides the [`APF`](crate::APF) enum

use num::Float;
use rand::prelude::*;
use rand_distr::{uniform::SampleUniform, Uniform};

use core::fmt::Debug;

/// Custom acceptance probability function
///
/// See why it's a `Box` [here](https://stackoverflow.com/a/59035722).
///
/// See the [`print`](Status#method.print) method for the signature explanation.
pub type Custom<'a, F, R> = Box<dyn Fn(F, F, &Uniform<F>, &mut R) -> bool + 'a>;

/// Acceptance probability function
pub enum APF<'a, F, R>
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
        f: Custom<'a, F, R>,
    },
}

impl<'a, F, R> APF<'a, F, R>
where
    F: Float + SampleUniform + Debug,
    R: Rng,
{
    /// Choose whether to accept the point
    ///
    /// # Arguments
    /// * `diff` --- Difference in the objective;
    /// * `t` --- Temperature;
    /// * `uni` -- Uniform[0, 1] distribution;
    /// * `rng` --- Random number generator.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn accept(&self, diff: F, t: F, uni: &Uniform<F>, rng: &mut R) -> bool {
        match *self {
            APF::Metropolis => {
                diff <= F::zero() || uni.sample(rng) < F::min(F::exp(-diff / t), F::one())
            }
            APF::Custom { ref f } => f(diff, t, uni, rng),
        }
    }
}
