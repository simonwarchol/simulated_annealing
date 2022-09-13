//! Provides the [`NeighbourMethod`](crate::NeighbourMethod) enum

use itertools::izip;
use num::Float;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};

use std::fmt::Debug;

use crate::{Bounds, Point};

/// Method of getting a random neighbour
pub enum Method<F, R, const N: usize>
where
    F: Float,
    StandardNormal: Distribution<F>,
    R: Rng,
{
    /// Get a neighbour in the vicinity of the current point
    /// by sampling a random normal distribution with the mean
    /// in that point and with the provided standard deviation
    Normal {
        /// Standard deviation
        sd: F,
    },
    /// Custom: choose your own!
    Custom {
        /// Custom function
        f: fn(p: &Point<F, N>, bounds: &Bounds<F, N>, rng: &mut R) -> Point<F, N>,
    },
}

impl<F, R, const N: usize> Method<F, R, N>
where
    F: Float + Debug,
    StandardNormal: Distribution<F>,
    R: Rng,
{
    /// Get a neighbour of the current point
    ///
    /// Arguments:
    /// * `p` --- Current point;
    /// * `bounds` --- Bounds of the parameter space;
    /// * `distribution` --- Distribution to sample from;
    /// * `rng` --- Random number generator.
    pub fn neighbour(&self, p: &Point<F, N>, bounds: &Bounds<F, N>, rng: &mut R) -> Point<F, N> {
        match self {
            Method::Normal { sd } => {
                let mut new_p = [F::zero(); N];
                // Generate a new point
                izip!(&mut new_p, p, bounds).for_each(|(np, &p, r)| {
                    // Create a normal distribution around the current coordinate
                    let d = Normal::new(p, *sd).unwrap();
                    // Sample from this distribution
                    let mut p = d.sample(rng);
                    // If the result is not in the range, repeat until it is
                    while !r.contains(&p) {
                        p = d.sample(rng);
                    }
                    // Save the new coordinate
                    *np = F::from(p).unwrap();
                });
                new_p
            }
            Method::Custom { f } => f(p, bounds, rng),
        }
    }
}
