//! This crate provides an implementation of the
//! [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
//! algorithm for approximating the global minimum of a given function.
//!
//! Choose the temperatures and the annealing schedule wisely:
//! this is your way of controlling the quality of the search
//! and how long you will have to wait. Note that the minimum
//! temperature must be reachable.
//!
//! References:
//! - Jason Brownlee, 2021, “[Simulated Annealing From Scratch in Python](https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/)”
//! - Mykel J. Kochenderfer, Tim A. Wheeler, 2019, “[Algorithms for Optimization](https://www.amazon.com/dp/0262039427)”
//! - Jonathan Woollett-Light, [`simple_optimization`](https://docs.rs/simple_optimization) crate
//!
//! Example:
//!
//! ```rust
//! use anyhow::{Context, Result};
//! use rand_xoshiro::rand_core::SeedableRng;
//! use simulated_annealing::{Bounds, NeighbourMethod, Point, Schedule, Status, APF, SA};
//!
//! // Define the objective function
//! fn f(p: &Point<f64, 1>) -> Result<f64> {
//!     let x = p[0];
//!     Ok(x.ln() * (x.sin() + x.cos()))
//! }
//! // Get the minimum (and the corresponding point)
//! let (m, p) = SA {
//!     // Objective function
//!     f,
//!     // Initial point
//!     p_0: &[2.],
//!     // Initial temperature
//!     t_0: 100_000.0,
//!     // Minimum temperature
//!     t_min: 1.0,
//!     // Bounds of the parameter space
//!     bounds: &[1.0..27.8],
//!     // Acceptance probability function
//!     apf: &APF::Metropolis,
//!     // Method of getting a random neighbour
//!     neighbour: &NeighbourMethod::Normal { sd: 5. },
//!     // Annealing schedule
//!     schedule: &Schedule::Fast,
//!     // Status function
//!     status: &mut Status::Periodic { nk: 1000 },
//!     // Random number generator
//!     rng: &mut rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1),
//! }
//! .findmin().with_context(|| "Couldn't find the global minimum")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod apf;
mod neighbour;
mod sa;
mod schedule;
mod status;
mod utils;

use core::ops::Range;

pub use apf::APF;
pub use neighbour::Method as NeighbourMethod;
pub use sa::SA;
pub use schedule::Schedule;
pub use status::{Custom as CustomStatus, Status};

/// Point in the parameter space
pub type Point<F, const N: usize> = [F; N];

/// Bounds of the parameter space
pub type Bounds<F, const N: usize> = [Range<F>; N];
