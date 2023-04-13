//! Provides the [`SA`](crate::SA) struct and the
//! [`minimum`](crate::SA#method.minimum) method

use anyhow::{Context, Result};
use num::Float;
use rand::prelude::*;
use rand_distr::{uniform::SampleUniform, Distribution, StandardNormal, Uniform};

use core::fmt::Debug;

use crate::{Bounds, NeighbourMethod, Point, Schedule, Status, APF};

/// Simulated annealing
pub struct SA<'a, 'apf, 'neighbour, 'schedule, 'status, F, R, FN, const N: usize>
where
    F: Float + SampleUniform + Debug,
    StandardNormal: Distribution<F>,
    R: Rng,
    FN: FnMut(&Point<F, N>) -> Result<F>,
{
    /// Objective function
    pub f: FN,
    /// Initial point
    pub p_0: &'a Point<F, N>,
    /// Initial temperature
    pub t_0: F,
    /// Minimum temperature
    pub t_min: F,
    /// Bounds of the parameter space
    pub bounds: &'a Bounds<F, N>,
    /// Acceptance probability function
    pub apf: &'a APF<'apf, F, R>,
    /// Method of getting a random neighbour
    pub neighbour: &'a NeighbourMethod<'neighbour, F, R, N>,
    /// Annealing schedule
    pub schedule: &'a Schedule<'schedule, F>,
    /// Status function
    pub status: &'a mut Status<'status, F, N>,
    /// Random number generator
    pub rng: &'a mut R,
}

impl<F, R, FN, const N: usize> SA<'_, '_, '_, '_, '_, F, R, FN, N>
where
    F: Float + SampleUniform + Debug,
    StandardNormal: Distribution<F>,
    R: Rng + SeedableRng,
    FN: FnMut(&Point<F, N>) -> Result<F>,
{
    /// Find the global minimum (and the corresponding point) of the objective function
    ///
    /// # Errors
    ///
    /// Will return `Err` if
    /// * Couldn't evaluate the objective function
    /// * Couldn't get a neighbour
    /// * Couldn't lower the temperature
    #[allow(clippy::arithmetic_side_effects)]
    pub fn findmin(&mut self) -> Result<(F, Point<F, N>)> {
        // Evaluate the objective function at the initial point and
        // save the initial values as the current working solution
        let mut p = *self.p_0;
        let mut f =
            (self.f)(self.p_0).with_context(|| "Couldn't evaluate the objective function")?;
        // Save the current working solution as the current best
        let mut best_p = p;
        let mut best_f = f;
        // Save the initial temperature as the current one
        let mut t = self.t_0;
        // Prepare the iterations counter
        let mut k = 1;
        // Prepare a Uniform[0, 1] distribution for the APF
        let uni = Uniform::new(F::zero(), F::one());
        // Search for the minimum of the objective function
        while t > self.t_min {
            // Get a neighbor
            let neighbour_p = self
                .neighbour
                .neighbour(&p, self.bounds, self.rng)
                .with_context(|| "Couldn't get a neighbor")?;
            // Evaluate the objective function
            let neighbour_f = (self.f)(&neighbour_p)
                .with_context(|| "Couldn't evaluate the objective function")?;
            // Compute the difference between the new and the current solutions
            let diff = neighbour_f - f;
            // If the new solution is accepted by the acceptance probability function,
            if self.apf.accept(diff, t, &uni, self.rng) {
                // Save it as the current solution
                p = neighbour_p;
                f = neighbour_f;
            }
            // If the new solution is the new best,
            if neighbour_f < best_f {
                // Save it as the new best
                best_p = neighbour_p;
                best_f = neighbour_f;
            }
            // Lower the temperature
            t = self
                .schedule
                .cool(k, t, self.t_0)
                .with_context(|| "Couldn't lower the temperature")?;
            // Print the status
            self.status.print(k, t, f, p, best_f, best_p);
            // Update the iterations counter
            k += 1;
        }
        Ok((best_f, best_p))
    }
}

#[cfg(test)]
use anyhow::bail;

#[test]
fn test() -> Result<()> {
    // Define the objective function
    #[allow(clippy::trivially_copy_pass_by_ref)]
    #[allow(clippy::unnecessary_wraps)]
    fn f(p: &Point<f64, 1>) -> Result<f64> {
        let x = p[0];
        Ok(x.ln() * (x.sin() + x.cos()))
    }
    // Get the minimum (and the corresponding point)
    let (m, p) = SA {
        f,
        p_0: &[2.],
        t_0: 100_000.0,
        t_min: 1.0,
        bounds: &[1.0..27.8],
        apf: &APF::Metropolis,
        neighbour: &NeighbourMethod::Normal { sd: 5. },
        schedule: &Schedule::Fast,
        status: &mut Status::Periodic { nk: 1000 },
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1),
    }
    .findmin()
    .with_context(|| "Couldn't find the global minimum")?;
    // Compare the result with the actual minimum
    let actual_p = [22.790_580_66];
    let actual_m = f(&actual_p).with_context(|| "Couldn't evaluate the objective function")?;
    if (p[0] - actual_p[0]).abs() >= 1e-4 {
        bail!(
            "The minimum point is incorrect: {} vs. {}",
            actual_p[0],
            p[0]
        );
    }
    if (m - actual_m).abs() >= 1e-9 {
        bail!("The minimum value is incorrect: {} vs. {}", actual_m, m);
    }
    Ok(())
}
