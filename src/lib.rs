//! This crate provides an implementation of the
//! [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
//! algorithm for approximating the global minimum of a given function.
//!
//! Choose the temperatures and the annealing schedule wisely:
//! this is your way of controlling how long you will have to wait.
//! Note that the minimum temperature must be reachable.
//!
//! References:
//! - Jason Brownlee, 2021, “[Simulated Annealing From Scratch in Python](https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/)”
//! - Mykel J. Kochenderfer, Tim A. Wheeler, 2019, “[Algorithms for Optimization](https://www.amazon.com/dp/0262039427)”
//! - Jonathan Woollett-Light, [`simple_optimization`](https://docs.rs/simple_optimization) crate

#[doc(hidden)]
mod apf;
#[doc(hidden)]
mod neighbour;
#[doc(hidden)]
mod sa;
#[doc(hidden)]
mod schedule;
#[doc(hidden)]
mod status;

use std::ops::Range;

pub use apf::APF;
pub use neighbour::Method as NeighbourMethod;
pub use sa::SA;
pub use schedule::Schedule;
pub use status::{Custom as CustomStatus, Status};

/// Point in the parameter space
pub type Point<F, const N: usize> = [F; N];

/// Bounds of the parameter space
pub type Bounds<F, const N: usize> = [Range<F>; N];
