"""Frozen LSMC regressions: device moments, host Cholesky and pivoted QR.

Scalar fits use DAL's normalized monomials, relative ridge and rank guard.
Two/three-feature fits use its ordered total-degree basis and scaled QR.
Only the small system (or, on rank loss, its design rows) reaches the host.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax.errors import InvalidSetting, script_error


@dataclass(frozen=True, slots=True)
class Regression:
    coefficients: tuple[float, ...] = ()
    means: tuple[float, ...] = (0.,)
    sigmas: tuple[float, ...] = (1.,)
    powers: tuple[tuple[int, ...], ...] = ()
    degree: int = 0
    count: int = 0
    rank: int = 0
    solver: str = "Constant"
    reason: str = ""
    fallback_reason: str = ""
    validation_mse: float | None = None

    def predict(self, x):
        """Jittable prediction; coefficients and normalization are passive."""
        if not self.coefficients:
            return self._empty_prediction(x)
        if len(self.coefficients) == 1:
            return jnp.full(self._prediction_shape(x), self.coefficients[0])
        z = (jnp.asarray(x)-jnp.asarray(self.means if len(self.means) > 1 else self.means[0])) / jnp.asarray(
            self.sigmas if len(self.means) > 1 else self.sigmas[0])
        if self.powers:
            return _design(z, self.powers) @ jnp.asarray(self.coefficients)
        value = jnp.zeros_like(z)
        for coefficient in reversed(self.coefficients):
            value = value*z+coefficient
        return value

    def _prediction_shape(self,x):
        return jnp.shape(x)[:-1] if len(self.means) > 1 else jnp.shape(x)

    def _empty_prediction(self,x):
        return jnp.zeros(jnp.shape(x)[:-1]) if len(self.means) > 1 else jnp.zeros_like(x)


def basis_powers(features: int, degree: int):
    if features == 1:
        return tuple((i,) for i in range(degree+1))
    if features == 2:
        return tuple((first, total-first) for total in range(degree+1) for first in range(total, -1, -1))
    return _three_feature_powers(degree)


def _three_feature_powers(degree):
    return tuple((first,second,total-first-second) for total in range(degree+1) for first in range(total,-1,-1)
                 for second in range(total-first,-1,-1))


def _design(z, powers):
    # Explicit products preserve DAL's first-to-last feature arithmetic.
    terms = []
    for term in powers:
        value = jnp.ones(z.shape[:-1], dtype=z.dtype)
        for i, exponent in enumerate(term):
            power = jnp.ones_like(value)
            for _ in range(exponent):
                power = power*z[..., i]
            value = value*power
        terms.append(value)
    return jnp.stack(terms, axis=-1)


@partial(jax.jit, static_argnames=("degree",))
def scalar_moments(x, y, included, degree):
    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
    count = jnp.sum(included)
    size = jnp.maximum(count, 1)
    mean = jnp.sum(jnp.where(included, x, 0.))/size
    deviations = jnp.where(included, x-mean, 0.)
    sigma = jnp.sqrt(jnp.sum(deviations*deviations)/size)
    floor = 1e-12*jnp.maximum(1., jnp.abs(mean))
    z = jnp.where(included, (x-mean)/jnp.maximum(sigma, floor), 0.)
    target = jnp.where(included, y, 0.)
    power = included.astype(jnp.float64)
    moments, rhs = [], []
    for i in range(2*degree+1):
        moments.append(jnp.sum(power))
        if i <= degree:
            rhs.append(jnp.sum(power*target))
        power = power*z
    return count, mean, sigma, floor, jnp.sum(target)/size, jnp.stack(moments), jnp.stack(rhs)


def _constant(fit, value, reason):
    return replace(fit, coefficients=(float(value),), powers=(), degree=0, rank=int(fit.count > 0),
                   solver="Constant", reason=reason, fallback_reason=reason)


def _qr_fit(design, targets):
    """DAL's scaled, column-pivoted, twice reorthogonalized Gram-Schmidt."""
    columns = np.array(design, dtype=np.float64, copy=True)
    n_basis = columns.shape[1]
    scales = np.linalg.norm(columns, axis=0)
    scales = np.where(scales == 0., 1., scales)
    columns /= scales
    permutation = np.arange(n_basis)
    upper = np.zeros((n_basis, n_basis))
    projection = np.zeros(n_basis)
    rank = 0
    for step in range(n_basis):
        norms = np.sum(columns[:, step:]**2, axis=0)
        pivot = step+int(np.argmax(norms))
        best = norms[pivot-step]
        if not np.isfinite(best) or best < 1e-20:
            break
        columns[:, [step, pivot]] = columns[:, [pivot, step]]
        scales[[step, pivot]] = scales[[pivot, step]]
        permutation[[step, pivot]] = permutation[[pivot, step]]
        upper[:step, [step, pivot]] = upper[:step, [pivot, step]]
        upper[step, step] = np.sqrt(best)
        columns[:, step] /= upper[step, step]
        projection[step] = columns[:, step] @ targets
        for term in range(step+1, n_basis):
            for _ in range(2):
                dot = columns[:, step] @ columns[:, term]
                upper[step, term] += dot
                columns[:, term] -= dot*columns[:, step]
        rank += 1
    coefficients = np.zeros(n_basis)
    if rank:
        solved = np.linalg.solve(upper[:rank, :rank], projection[:rank])
        coefficients[permutation[:rank]] = solved/scales[:rank]
    return coefficients, rank


def _gram_reason(gram):
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        return "GramRankLoss"
    scale = np.sqrt(diagonal)
    scaled = gram/scale[:, None]/scale[None, :]
    try:
        lower = np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError:
        return "GramRankLoss"
    if np.any(np.diag(lower)**2 <= 1e-12):
        return "GramRankLoss"
    return "GramScale" if np.max(diagonal)/np.min(diagonal) > 1e12 else ""


def _scalar_fit(x, y, included, degree):
    count, mean, sigma, floor, constant, moments, rhs = jax.device_get(scalar_moments(x, y, included, degree))
    fit = Regression(means=(float(mean),), sigmas=(float(max(sigma, floor)),), count=int(count))
    fallback = _scalar_guard(fit,constant,sigma,floor,degree)
    if fallback is not None:
        return fallback
    indices = np.arange(degree+1)
    gram = moments[indices[:, None]+indices]
    reason = _gram_reason(gram)
    if not reason:
        regularized = gram+np.diag(np.diag(gram)*1e-12)
        lower = np.linalg.cholesky(regularized)
        coefficients = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
        return replace(fit, coefficients=tuple(coefficients), degree=degree, rank=degree+1, solver="MomentsCholesky")
    x_host, y_host, mask = jax.device_get((x, y, included))
    z = (np.asarray(x_host)[mask]-mean)/fit.sigmas[0]
    effective_rank = degree+1
    for candidate in range(degree, 0, -1):
        coefficients, rank = _qr_fit(np.vander(z, candidate+1, increasing=True), np.asarray(y_host)[mask])
        effective_rank = min(effective_rank, rank)
        if rank == candidate+1:
            return replace(fit, coefficients=tuple(coefficients), degree=candidate, rank=effective_rank,
                           solver="PivotedQR", fallback_reason=reason if candidate == degree else "RankDeficient")
    return _constant(fit, constant, "IllConditioned")


def _scalar_guard(fit,constant,sigma,floor,degree):
    if fit.count == 0:
        return _constant(fit,constant,"ConditionPathsBelowMin")
    if not all(np.isfinite(v) for v in (*fit.means,sigma,constant)):
        raise script_error("InvalidRegressionInput: non-finite included regressor or target")
    if sigma < floor:
        return _constant(fit,constant,"SigmaFloor")
    if fit.count < 10*(degree+1):
        return _constant(fit,constant,"ConditionPathsBelowMin")
    return None


def _multi_fit(x, y, included, degree):
    x, y, mask = jax.device_get((x, y, included))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    count, features = x.shape
    fit = Regression(means=(0.,)*features, sigmas=(1.,)*features, count=count)
    if count == 0:
        return _constant(fit, 0., "ConditionPathsBelowMin")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise script_error("InvalidRegressionInput: non-finite included feature or target")
    fit,constant,below = _multi_normalization(fit,x,y)
    for candidate in range(degree,0,-1):
        powers = basis_powers(features,candidate)
        if count < 10*len(powers):
            continue
        design = np.asarray(_design(jnp.asarray((x-fit.means)/fit.sigmas),powers))
        coefficients,rank = _qr_fit(design,y)
        if rank > 1:
            return _multi_candidate(fit,coefficients,powers,rank,candidate,degree,below)
    return _multi_constant(fit,constant,below)


def _multi_normalization(fit,x,y):
    # Native multivariate normalization uses Welford's path-order moments.
    count,features = x.shape
    means, m2, constant = np.zeros(features), np.zeros(features), 0.
    for i, (row, target) in enumerate(zip(x, y), 1):
        constant += (target-constant)/i
        delta = row-means
        means += delta/i
        m2 += delta*(row-means)
    sigma = np.sqrt(np.maximum(m2/count, 0.))
    below = sigma < 1e-10*np.maximum(1., np.abs(means))
    sigma = np.where(below, 1., sigma)
    fit = replace(fit, means=tuple(means), sigmas=tuple(sigma))
    return fit,constant,below


def _multi_candidate(fit,coefficients,powers,rank,candidate,degree,below):
    reason = "SigmaFloor" if np.any(below) else "RankDeficient"
    fallback = reason if rank < len(powers) else ("ConditionPathsBelowMin" if candidate < degree else "")
    return replace(fit,coefficients=tuple(coefficients),powers=powers,degree=candidate,rank=rank,
                   solver="PivotedQR",fallback_reason=fallback)


def _multi_constant(fit,constant,below):
    reason = "ConditionPathsBelowMin" if fit.count < 10*(len(fit.means)+1) else ("SigmaFloor" if np.any(below) else "IllConditioned")
    return _constant(fit, constant, reason)


def solve_regression(x, targets, included, degree=3):
    """Fit an immutable scalar or total-degree multivariate policy."""
    x, targets, included = jnp.asarray(x, jnp.float64), jnp.asarray(targets, jnp.float64), jnp.asarray(included, bool)
    features = 1 if x.ndim == 1 else x.shape[-1]
    if not 1 <= degree <= (8 if features == 1 else 3):
        raise InvalidSetting("LSMC basis degree must be 1..8 (1..3 for multiple features)")
    _validate_regression_vectors(x,targets,included)
    if features not in (1, 2, 3):
        raise script_error("InvalidLsmcFeatureBudget: expected one to three features")
    return _scalar_fit(x.reshape(-1), targets, included, degree) if features == 1 else _multi_fit(x, targets, included, degree)


def _validate_regression_vectors(x,targets,included):
    if targets.ndim != 1 or included.shape != targets.shape or x.shape[0] != targets.shape[0]:
        raise script_error("InvalidRegressionInput: mismatched regression vectors")


def select_regression(x, targets, included, degree, validation=None):
    """Select the smallest candidate within one standard error of the best loss."""
    if validation is None or not np.any(np.asarray(validation[2])):
        return solve_regression(x, targets, included, degree)
    fits = tuple(solve_regression(x, targets, included, d) for d in range(1, degree+1))
    vx, vy, mask = validation
    predictions = jax.vmap(lambda i: jnp.stack(tuple(fit.predict(vx) for fit in fits))[i])(jnp.arange(degree))
    losses = jnp.where(jnp.asarray(mask)[None, :], (predictions-vy)**2, 0.)
    count = jnp.sum(mask)
    mse = jnp.sum(losses, axis=1)/count
    se = jnp.sqrt(jnp.maximum(0., jnp.sum(losses**2, axis=1)/count-mse**2)/count)
    mse, se = jax.device_get((mse, se))
    best = int(np.argmin(mse))
    threshold = mse[best]+se[best]+1e-10*max(1., mse[best])
    chosen = next((i for i, loss in enumerate(mse) if np.isfinite(loss) and loss <= threshold), 0)
    return replace(fits[chosen], validation_mse=float(mse[chosen]))
