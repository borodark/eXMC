"""The seven race models, PyMC side (docs/PYMC_RACE_PLAN.md).

Specs are phd.git `benchmark/benchmark_pymc.py` @ c579dba, verbatim, with ONE
deliberate change, recorded here because it is a change of model:

  sv: `GaussianRandomWalk` gets `init_dist=pm.Normal.dist(0, sigma)`.
      The February builder passed no init_dist. PyMC then defaults the first
      step to Normal(0, 100) (and PyMC 6 warns), while exmc's GaussianRandomWalk
      has x[0] ~ Normal(0, sigma). So the February SV race compared two
      different models. The parity gate (parity.py / parity.exs) is what
      found it; this makes the PyMC side the model exmc implements.

`build(name, data)` returns `(model, var_names)`: the free RVs whose draws are
scored, in the order the results files use.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MODELS = ["simple", "medium", "stress", "eight_schools", "funnel", "logistic", "sv"]


def build(name, data):
    return _BUILDERS[name](data)


def _simple(data):
    y = np.array(data["simple"]["y"], dtype=float)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.Exponential("sigma", lam=1)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    return model, ["mu", "sigma"]


def _medium(data):
    y_a = np.array(data["medium"]["y_a"], dtype=float)
    y_b = np.array(data["medium"]["y_b"], dtype=float)
    with pm.Model() as model:
        mu_global = pm.Normal("mu_global", mu=0, sigma=10)
        sigma_global = pm.Exponential("sigma_global", lam=1)
        alpha = pm.Normal("alpha", mu=mu_global, sigma=sigma_global)
        beta = pm.Normal("beta", mu=mu_global, sigma=sigma_global)
        sigma_obs = pm.Exponential("sigma_obs", lam=2)
        pm.Normal("y_a_obs", mu=alpha, sigma=sigma_obs, observed=y_a)
        pm.Normal("y_b_obs", mu=beta, sigma=sigma_obs, observed=y_b)
    return model, ["mu_global", "sigma_global", "alpha", "beta", "sigma_obs"]


def _stress(data):
    ys = [np.array(data["stress"][f"y_{j}"], dtype=float) for j in (1, 2, 3)]
    with pm.Model() as model:
        mu_pop = pm.Normal("mu_pop", mu=0, sigma=10)
        sigma_pop = pm.Exponential("sigma_pop", lam=0.5)
        groups = [pm.Normal(f"group_{j}", mu=mu_pop, sigma=sigma_pop) for j in (1, 2, 3)]
        noises = [pm.Exponential(f"noise_{j}", lam=1) for j in (1, 2, 3)]
        for j in range(3):
            pm.Normal(f"y{j + 1}_obs", mu=groups[j], sigma=noises[j], observed=ys[j])
    return model, ["mu_pop", "sigma_pop", "group_1", "group_2", "group_3",
                   "noise_1", "noise_2", "noise_3"]


def _eight_schools(data):
    es = data["eight_schools"]
    y = np.array(es["y"], dtype=float)
    sigma = np.array(es["sigma"], dtype=float)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfNormal("tau", sigma=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("y_obs", mu=theta, sigma=sigma, observed=y)
    return model, ["mu", "tau", "theta"]


def _funnel(data):
    with pm.Model() as model:
        y = pm.Normal("y", mu=0, sigma=3)
        pm.Normal("x", mu=0, sigma=pt.exp(y / 2), shape=9)
    return model, ["y", "x"]


def _logistic(data):
    lg = data["logistic"]
    X = np.array(lg["X"], dtype=float)
    y = np.array(lg["y"])
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])
        pm.Bernoulli("y_obs", logit_p=alpha + pt.dot(X, beta), observed=y)
    return model, ["alpha", "beta"]


def _sv(data):
    sv = data["sv"]
    returns = np.array(sv["returns"], dtype=float)
    T = sv["T"]
    with pm.Model() as model:
        sigma = pm.Exponential("sigma", lam=50)
        nu = pm.Exponential("nu", lam=0.1)
        s = pm.GaussianRandomWalk("s", sigma=sigma, init_dist=pm.Normal.dist(0, sigma), shape=T)
        pm.StudentT("r_obs", nu=nu, mu=0, sigma=pm.math.exp(s), observed=returns)
    return model, ["sigma", "nu", "s"]


_BUILDERS = {
    "simple": _simple,
    "medium": _medium,
    "stress": _stress,
    "eight_schools": _eight_schools,
    "funnel": _funnel,
    "logistic": _logistic,
    "sv": _sv,
}
