"""Fixed-shape regression for vmapped LSMC policy bumps.

The ordinary driver uses host small-system solves. Retraining multiple bumped
inputs keeps the same moments/rank guards on device, including pivoted QR.
All normalization and regression arithmetic uses float64.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from dal_jax.mc.regression import basis_powers, _design


class Fit(NamedTuple):
    coefficients: object
    means: object
    sigmas: object
    degree: object
    rank: object

    def predict(self, x):
        constant = jnp.all(self.coefficients[1:] == 0.)
        z = (jnp.where(constant, self.means, x)-self.means)/self.sigmas
        if x.shape[-1] == 1:
            value = jnp.zeros(x.shape[:-1], dtype=x.dtype)
            for coefficient in reversed(jnp.unstack(self.coefficients)):
                value = value*z[..., 0]+coefficient
            return value
        degree = next(d for d in range(4) if len(basis_powers(x.shape[-1], d)) == self.coefficients.shape[-1])
        return _design(z, basis_powers(x.shape[-1], degree)) @ self.coefficients


def pivoted_qr(design, targets):
    """Fixed-size counterpart of DAL's column-scaled two-pass QR."""
    basis = design.shape[-1]
    scales = jnp.sqrt(jnp.sum(design*design, axis=0))
    scales = jnp.where(scales == 0., 1., scales)
    initial = (design/scales, scales, jnp.arange(basis), jnp.zeros((basis, basis)), jnp.zeros(basis), jnp.asarray(0, jnp.int32), jnp.asarray(True))
    positions = jnp.arange(basis)

    def step(index, carry):
        columns, norms, permutation, upper, projection, rank, alive = carry
        squared = jnp.sum(columns*columns, axis=0)
        pivot = jnp.argmax(jnp.where(positions >= index, squared, -1.))
        best = squared[pivot]
        alive = alive & jnp.isfinite(best) & (best >= 1e-20)
        swaps = jnp.where(positions == index, pivot, jnp.where(positions == pivot, index, positions))
        columns, norms, permutation = columns[:, swaps], norms[swaps], permutation[swaps]
        upper = jnp.where(positions[:, None] < index, upper[:, swaps], upper)
        diagonal = jnp.sqrt(jnp.where(alive, best, 1.))
        q = columns[:, index]/diagonal
        columns = columns.at[:, index].set(q)
        upper = upper.at[index, index].set(diagonal)
        projection = projection.at[index].set(jnp.where(alive, q @ targets, 0.))
        for _ in range(2):
            dots = jnp.where(positions > index, q @ columns, 0.)
            upper = upper.at[index, :].add(dots)
            columns = columns-q[:, None]*dots[None, :]
        return columns, norms, permutation, upper, projection, rank+alive.astype(jnp.int32), alive

    _, scales, permutation, upper, projection, rank, _ = jax.lax.fori_loop(0, basis, step, initial)
    independent = positions < rank
    upper = jnp.where(independent[:, None] & independent[None, :], upper, jnp.eye(basis))
    pivoted = solve_triangular(upper, jnp.where(independent, projection, 0.), lower=False)
    coefficients = jnp.zeros(basis).at[permutation].set(pivoted/scales)
    return coefficients, rank


def _normalization(x, y, included, features):
    count = jnp.sum(included)
    size = jnp.maximum(count, 1)
    clean_x = jnp.where(included[:, None], x, 0.)
    clean_y = jnp.where(included, y, 0.)
    means = jnp.sum(clean_x, axis=0)/size
    deviations = jnp.where(included[:, None], x-means, 0.)
    sigma = jnp.sqrt(jnp.sum(deviations*deviations, axis=0)/size)
    floor = (1e-12 if features == 1 else 1e-10)*jnp.maximum(1., jnp.abs(means))
    sigmas = jnp.maximum(sigma, floor) if features == 1 else jnp.where(sigma < floor, 1., sigma)
    return count, means, sigmas, sigma < floor, jnp.sum(clean_y)/size, clean_y


def _constant_fit(width, means, sigmas, constant, count):
    return Fit(jnp.zeros(width).at[0].set(constant), means, sigmas, jnp.asarray(0, jnp.int32), (count > 0).astype(jnp.int32))


def _scalar_qr(z, y, included, degree, constant):
    selected = jnp.zeros(degree+1).at[0].set(constant)
    selected_degree, effective_rank = jnp.asarray(0, jnp.int32), jnp.asarray(degree+1, jnp.int32)
    for candidate in range(degree, 0, -1):
        powers = _design(z, basis_powers(1, candidate))
        design = jnp.where(included[:, None], powers, 0.)
        coefficients, rank = pivoted_qr(design, y)
        take = (selected_degree == 0) & (rank == candidate+1)
        selected = jnp.where(take, jnp.pad(coefficients, (0, degree-candidate)), selected)
        effective_rank = jnp.where(selected_degree == 0, jnp.minimum(effective_rank, rank), effective_rank)
        selected_degree = jnp.where(take, candidate, selected_degree)
    return selected, selected_degree, effective_rank


def _scalar_fit(x, y, included, degree, stats):
    count, means, sigmas, below, constant, clean_y = stats
    fallback = _constant_fit(degree+1, means, sigmas, constant, count)
    z = jnp.where(included[:, None], (x-means)/sigmas, 0.)
    powers = _design(z, basis_powers(1, 2*degree))
    moments = jnp.sum(jnp.where(included[:, None], powers, 0.), axis=0)
    rhs = jnp.sum(powers[:, :degree+1]*clean_y[:, None], axis=0)
    ids = jnp.arange(degree+1)
    gram = moments[ids[:, None]+ids[None, :]]
    diagonal = jnp.diag(gram)
    scales = jnp.sqrt(jnp.maximum(diagonal, 1e-300))
    lower = jnp.linalg.cholesky(gram/scales[:, None]/scales[None, :])
    bad = jnp.any(~jnp.isfinite(lower)) | jnp.any(jnp.diag(lower)**2 <= 1e-12) | (jnp.max(diagonal)/jnp.min(diagonal) > 1e12)

    def cholesky(_):
        ridge = gram+jnp.diag(diagonal*1e-12)
        factor = jnp.linalg.cholesky(ridge)
        coefficients = solve_triangular(factor.T, solve_triangular(factor, rhs, lower=True), lower=False)
        return Fit(coefficients, means, sigmas, jnp.asarray(degree, jnp.int32), jnp.asarray(degree+1, jnp.int32))

    def qr(_):
        coefficients, used, rank = _scalar_qr(z, clean_y, included, degree, constant)
        return Fit(coefficients, means, sigmas, used, rank)
    return jax.lax.cond((count >= 10*(degree+1)) & ~jnp.any(below),
                        lambda _: jax.lax.cond(bad, qr, cholesky, None), lambda _: fallback, None)


def _multi_fit(x, y, included, degree, stats):
    count, means, sigmas, _, constant, clean_y = stats
    features = x.shape[-1]
    width = len(basis_powers(features, degree))
    selected = _constant_fit(width, means, sigmas, constant, count)
    z = jnp.where(included[:, None], (x-means)/sigmas, 0.)
    for candidate in range(degree, 0, -1):
        powers = basis_powers(features, candidate)
        design = jnp.where(included[:, None], _design(z, powers), 0.)
        coefficients, rank = pivoted_qr(design, clean_y)
        take = (selected.degree == 0) & (count >= 10*len(powers)) & (rank > 1)
        selected = Fit(jnp.where(take, jnp.pad(coefficients, (0, width-len(powers))), selected.coefficients), means, sigmas,
                       jnp.where(take, candidate, selected.degree), jnp.where(take, rank, selected.rank))
    return selected


def solve_device(x, targets, included, degree):
    """Pure, vmappable fixed-shape fit; call host validation at the API boundary."""
    x, targets = jnp.asarray(x, jnp.float64), jnp.asarray(targets, jnp.float64)
    stats = _normalization(x, targets, included, x.shape[-1])
    return _scalar_fit(x, targets, included, degree, stats) if x.shape[-1] == 1 else _multi_fit(x, targets, included, degree, stats)


def select_device(x, targets, included, degree, validation=None):
    if validation is None:
        return solve_device(x, targets, included, degree)
    vx, vy, mask = validation
    width = len(basis_powers(x.shape[-1], degree))
    fits = tuple(solve_device(x, targets, included, candidate) for candidate in range(1, degree+1))
    predictions = jnp.stack(tuple(fit.predict(vx) for fit in fits))
    losses = jnp.where(mask[None, :], (predictions-vy)**2, 0.)
    count = jnp.sum(mask)
    size = jnp.maximum(count, 1)
    mse = jnp.sum(losses, axis=1)/size
    se = jnp.sqrt(jnp.maximum(0., jnp.sum(losses**2, axis=1)/size-mse*mse)/size)
    best = jnp.argmin(mse)
    threshold = mse[best]+se[best]+1e-10*jnp.maximum(1., mse[best])
    chosen = jnp.argmax(mse <= threshold)
    stacked = jax.tree.map(lambda *fields: jnp.stack(fields), *tuple(fit._replace(coefficients=jnp.pad(fit.coefficients,
                             (0, width-fit.coefficients.shape[0]))) for fit in fits))
    selected = jax.tree.map(lambda field: field[chosen], stacked)
    default = solve_device(x, targets, included, degree)
    return jax.tree.map(lambda chosen_field, default_field: jnp.where(count > 0, chosen_field, default_field), selected, default)
