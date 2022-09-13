//! Provides the [`Schedule`](crate::Schedule) enum

use num::Float;
use numeric_literals::replace_float_literals;

use std::fmt::Debug;

/// Annealing schedule
pub enum Schedule<F: Float> {
    /// Logarithmic:
    ///
    /// $ t^{(k)} = t^{(1)} \ln(2) / \ln(k + 1) $
    Logarithmic,
    /// Exponential:
    ///
    /// $ t^{(k+1)} = \gamma t^{(k)} \\; \text{for} \\; \gamma \in (0, 1) $
    Exponential {
        /// Exponential parameter $ \gamma $
        gamma: F,
    },
    /// Fast:
    ///
    /// $ t^{(k)} = t^{(1)} / k $
    Fast,
    /// Custom: choose your own!
    Custom {
        /// Custom function
        f: fn(k: usize, t: F, t_0: F) -> F,
    },
}

impl<F: Float + Debug> Schedule<F> {
    /// Lower the temperature
    ///
    /// Arguments:
    /// * `k` --- Index of the iteration;
    /// * `t` --- Temperature,
    /// * `t_0` --- Initial temperature.
    #[replace_float_literals(F::from(literal).unwrap())]
    pub fn cool(&self, k: usize, t: F, t_0: F) -> F {
        match self {
            Schedule::Logarithmic => t_0 * F::ln(2.) / F::ln(F::from(k + 1).unwrap()),
            Schedule::Exponential { gamma } => *gamma * t,
            Schedule::Fast => t_0 / F::from(k).unwrap(),
            Schedule::Custom { f } => f(k, t, t_0),
        }
    }
}
