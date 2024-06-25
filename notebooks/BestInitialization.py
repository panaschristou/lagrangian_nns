# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from matplotlib import pyplot as plt

from jax import grad, vmap
from jax.example_libraries import optimizers

from lnn.experiment_dblpend.lnn import raw_lagrangian_eom
from lnn.experiment_dblpend.data import get_trajectory_analytic
from lnn.experiment_dblpend.physics import analytical_fn
from lnn.hyperopt import HyperparameterSearch
from lnn.hyperopt.HyperparameterSearch import learned_dynamics
from lnn.hyperopt.HyperparameterSearch import extended_mlp


# %% [markdown]
# ## Set up LNN:

# %%
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


# %%

# %%

# %%
vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend="cpu")(
    jax.vmap(
        partial(
            get_trajectory_analytic,
        ),
        (0, None),
        0,
    )
)

# %%

# %% [markdown]
# ## Here are our model parameters

# %%
args = ObjectView(
    {
        "dataset_size": 200,
        "fps": 10,
        "samples": 100,
        "num_epochs": 80000,
        "seed": 0,
        "loss": "l1",
        "act": "softplus",
        "hidden_dim": 30,
        "output_dim": 1,
        "layers": 3,
        "n_updates": 1,
        "lr": 0.001,
        "lr2": 2e-05,
        "dt": 0.1,
        "model": "gln",
        "batch_size": 68,
        "l2reg": 5.7e-07,
    }
)
# args = loaded['args']
rng = jax.random.PRNGKey(args.seed)

# %%

# %%

# %%

# %%
vfnc = jax.jit(jax.vmap(analytical_fn, 0, 0))
vget = partial(jax.jit, backend="cpu")(
    jax.vmap(
        partial(
            get_trajectory_analytic,
        ),
        (0, None),
        0,
    )
)

batch = 60


@jax.jit
def get_derivative_dataset(rng):
    # randomly sample inputs

    y0 = jnp.concatenate(
        [
            jax.random.uniform(rng, (batch, 2)) * 2.0 * np.pi,
            (jax.random.uniform(rng + 1, (batch, 2)) - 0.5) * 10 * 2,
        ],
        axis=1,
    )

    return y0, vfnc(y0)


# %%
best_params = None
best_loss = np.inf

# %%
init_random_params, nn_forward_fn = extended_mlp(args)

HyperparameterSearch.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng + 1, (-1, 4))
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init(init_params)
# train(args, model, data, rng);

# %% [markdown]
# Current std:

# %%

# %%
HyperparameterSearch.nn_forward_fn = nn_forward_fn

# %% [markdown]
# ## Let's score the qdotdot output over normally distributed input for 256 batch size:

# %%
normal = True
n = 256


@jax.jit
def custom_init(stds, rng2):
    new_params = []
    i = 0
    for l1 in init_params:
        if (len(l1)) == 0:
            new_params.append(())
            continue
        new_l1 = []
        for l2 in l1:
            if len(l2.shape) == 1:
                new_l1.append(jnp.zeros_like(l2))
            else:
                if normal:
                    new_l1.append(jax.random.normal(rng2, l2.shape) * stds[i])
                #                     n1 = l2.shape[0]
                #                     n2 = l2.shape[1]
                #                     power = stds[0]
                #                     base_scale = stds[1]
                #                     s = base_scale/(n1+n2)**power
                #                     new_l1.append(jax.random.normal(rng2, l2.shape)*s)
                else:
                    new_l1.append(
                        jax.random.uniform(rng2, l2.shape, minval=-0.5, maxval=0.5)
                        * stds[i]
                    )
                rng2 += 1
                i += 1

        new_params.append(new_l1)

    return new_params


@jax.jit
def j_score_init(stds, rng2):
    new_params = custom_init(stds, rng2)

    rand_input = jax.random.normal(rng2, [n, 4])
    rng2 += 1

    outputs = jax.vmap(partial(raw_lagrangian_eom, learned_dynamics(new_params)))(
        rand_input
    )[:, 2:]

    # KL-divergence to mu=0, std=1:
    mu = jnp.average(outputs, axis=0)
    std = jnp.std(outputs, axis=0)

    KL = jnp.sum((mu**2 + std**2 - 1) / 2.0 - jnp.log(std))

    def total_output(p):
        return vmap(partial(raw_lagrangian_eom, learned_dynamics(p)))(rand_input).sum()

    d_params = grad(total_output)(new_params)

    for l1 in d_params:
        if (len(l1)) == 0:
            continue
        for l2 in l1:
            if len(l2.shape) == 1:
                continue

            mu = jnp.average(l2)
            std = jnp.std(l2)
            KL += (mu**2 + std**2 - 1) / 2.0 - jnp.log(std)

    # HACK
    #     KL += jnp.sum(stds**2)
    return jnp.log10(KL)


# %%
cur_std = jnp.array([0.01] * (args.layers + 1))

rng2 = jax.random.PRNGKey(0)

# %%
j_score_init(cur_std, rng2)

# %%
# @jax.jit

vv = jax.jit(vmap(j_score_init, (None, 0), 0))

rng2 = jax.random.PRNGKey(0)


def score_init(stds):
    global rng2
    stds = jnp.array(stds)
    stds = jnp.exp(stds)
    q75, q50, q25 = np.percentile(
        vv(stds, jax.random.split(rng2, num=10)), [75, 50, 25]
    )
    rng2 += 30

    return q50, q75 - q25


# %%
score_init(cur_std)

# %%
# from bayes_opt import BayesianOptimization

# # Bounded region of parameter space
pbounds = {"s%d" % (i,): (-15, 15) for i in range(len(cur_std))}


# %%
def bb(**kwargs):
    out, std = score_init(
        [kwargs[q] for q in ["s%d" % (i,) for i in range(len(cur_std))]]
    )
    #     if out is None or not out > -30:
    #         return -30.0
    return -out, std


# %% [markdown]
# Let's fit the best distribution:

# %% [markdown]
# # Let's redo that with Bayes:

# %% [markdown]
# # Bayesian:

# %% [markdown]
# # Old stuff:

# %%
from hyperopt import hp, fmin, tpe, Trials


def run_trial(args):
    loss, std = bb(**args)
    if loss == np.nan:
        return {
            "status": "fail",  # or 'fail' if nan loss
            "loss": np.inf,
        }

    return {
        "status": "ok",  # or 'fail' if nan loss
        "loss": -loss,
        "loss_variance": std,
    }


# TODO: Declare your hyperparameter priors here:
space = {
    **{"s%d" % (i,): hp.normal("s%d" % (i,), -2, 5) for i in range(len(cur_std) - 1)},
    **{"s%d" % (len(cur_std) - 1,): hp.normal("s%d" % (len(cur_std) - 1,), 3, 8)},
}

# %%
trials = Trials()

# %%
best = fmin(
    run_trial, space=space, algo=tpe.suggest, max_evals=5000, trials=trials, verbose=1
)


# %%
def k(t):
    if "loss" not in t["result"]:
        return np.inf
    return t["result"]["loss"]


sorted_trials = sorted(trials.trials, key=k)
len(trials.trials)

# %%
q = np.array(
    [
        [s["misc"]["vals"]["s%d" % (i,)][0] for i in range(len(cur_std))]
        for s in sorted_trials[:100]
    ]
)
q[0]

# %% [markdown]
# ## 4 layers, 1000 hidden: {(4, 1000), (1000, 1000), (1000, 1000), (1000, 1)}
#
# ## median top 10/2000: array([-1.47842217, -4.37217279, -3.37083752, 11.13480387])
#
# (unconverged)
#
# ## 4 layers, 100 hidden:  {(4, 100), (100, 100), (100, 100), (100, 1)}
#
# ## median top 30/5000: array([-1.70680816, -2.40340615, -2.17201716, 10.55268474])
#
# (unconverged)
#
# ## 3 layers, 100 hidden:
#
# ## median top 100/7000: array([-1.69875614, -2.74589338,  3.75818009])
#
# (unverged converged)
#
# ## 3 layers, 30 hidden:

# %% [markdown]
# # Use Eureqa to get the scalings!

# %%
simple_data = np.array(
    [
        [t["misc"]["vals"]["s%d" % (i,)][0] for i in range(len(cur_std))]
        + [t["result"]["loss"]]
        for t in trials.trials
        if "loss" in t["result"] and np.isfinite(t["result"]["loss"])
    ]
)

# %%
# np.save('sdata.npy', simple_data)

# %%
from sklearn.gaussian_process import GaussianProcessRegressor

# %%
gp = GaussianProcessRegressor(alpha=3, n_restarts_optimizer=20, normalize_y=True)

# %%
simple_data[:, -1].min()

# %%
gp.fit(simple_data[:, :-1], simple_data[:, -1])

# %%
args.layers + 1, args.hidden_dim, q[gp.predict(q).argmin()]

# %% [markdown]
# # New runs with noise added:
#
# ## layers, hidden, log(std): (4, 30, array([-2.12770715, -1.99764457, -1.29472256,  6.1514019 ]))

# %%

# %%

# %%

# %% [markdown]
# ## predicted with GP for 3 layers, 100 hidden, {(4, 100), (100, 100), (100, 1)}
#
# array([-1.95669793, -2.39555616,  1.92755129])
#
# ## predicted with GP for 3 layers, 50 hidden, {(4, 50), (50, 50), (50, 1)}
#
# array([-1.77223004, -3.2154843 , 10.38542243])
#
#
# ## predicted with GP for 3 layers, 30 hidden:
#
# array([-1.47298021, -4.10931435,  2.60899782])
#
#
#
#
#

# %%
plt.scatter(simple_data[:, 0], simple_data[:, 1], c=simple_data[:, -1])
# plt.ylim(-5, 2)
# plt.xlim(-5, 2)

# %%
# num= 50
# x = np.linspace(-12, 2, num=num)
# y = np.linspace(-12, 2, num=num)
# X,Y = np.meshgrid(x, y) # grid of point
# Z = gp.predict(np.stack((X.ravel(), Y.ravel()), axis=1)).reshape(*[num]*2) # evaluation of the function on the grid

# im = plt.imshow(np.log10(Z), cmap='viridis', extent=[-12, 2, -12, 2], origin='lower')
# plt.colorbar()

# %%

# %%

# %%

# %%
from bayes_opt import BayesianOptimization

# %%
optimizer = BayesianOptimization(
    f=bb,
    pbounds=pbounds,
    random_state=1,
)

# %%
optimizer.maximize(
    init_points=4 ** len(cur_std),
    n_iter=300 + 4 ** len(cur_std),
    alpha=1e-2,
    normalize_y=True,
)

# %%
optimizer.max

# %%
for l1 in init_params:
    if (len(l1)) == 0:
        continue
    for l2 in l1:
        print(l2.shape)

# %%
simple_data = np.array(
    [
        [t["params"]["s%d" % (i,)] for i in range(len(cur_std))] + [t["target"]]
        for t in optimizer.res
    ]
)

# %%
np.save(
    "hidden={}_layers={}_results.npy".format(args.hidden_dim, args.layers), simple_data
)

# %%
from sklearn.gaussian_process import GaussianProcessRegressor

# %%
gp = GaussianProcessRegressor(alpha=1e-2, n_restarts_optimizer=20, normalize_y=True)

# %%
# simple_data[:, -1].min()

# %%
gp.fit(simple_data[:, :-1], simple_data[:, -1])

# %%
simple_data[:, :-1][gp.predict(simple_data[:, :-1]).argmin()]
