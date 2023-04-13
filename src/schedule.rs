//! Provides the [`Schedule`](crate::Schedule) enum

use anyhow::Result;
use num::Float;

use core::fmt::Debug;

use crate::utils::cast;

/// Custom annealing schedule
///
/// See why it's a `Box` [here](https://stackoverflow.com/a/59035722).
///
/// See the [`cool`](Schedule#method.cool) method for the signature explanation.
pub type Custom<'a, F> = Box<dyn Fn(usize, F, F) -> Result<F> + 'a>;

/// Annealing schedule
pub enum Schedule<'a, F: Float> {
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
        f: Custom<'a, F>,
    },
}

impl<'a, F: Float + Debug> Schedule<'a, F> {
    /// Lower the temperature
    ///
    /// # Arguments
    /// * `k` --- Index of the iteration;
    /// * `t` --- Temperature,
    /// * `t_0` --- Initial temperature.
    ///
    /// # Errors
    ///
    /// Will return `Err` if
    /// * Logarithmic, Fast: couldn't cast a number to a generic floating-point number
    /// * Custom function returned `Err`
    #[allow(clippy::arithmetic_side_effects)]
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::unwrap_in_result)]
    #[allow(clippy::unwrap_used)]
    pub fn cool(&self, k: usize, t: F, t_0: F) -> Result<F> {
        match *self {
            Schedule::Logarithmic => Ok(t_0 * F::ln(F::from(2.).unwrap()) / F::ln(cast(k + 1)?)),
            Schedule::Exponential { gamma } => Ok(gamma * t),
            Schedule::Fast => Ok(t_0 / cast(k)?),
            Schedule::Custom { ref f } => f(k, t, t_0),
        }
    }
}
