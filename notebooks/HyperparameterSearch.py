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
#     display_name: main2
#     language: python
#     name: main2
# ---

# %%
import importlib
from copy import deepcopy as copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers
from jax.tree_util import tree_flatten
from lnn.experiment_dblpend.data import get_trajectory_analytic
from lnn.experiment_dblpend.physics import analytical_fn
from lnn.hyperopt import HyperparameterSearch
from lnn.hyperopt.HyperparameterSearch import (extended_mlp, make_loss,
                                               new_get_dataset, train)


# %%
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


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
# 0.29830917716026306 {'act': [4],
# 'batch_size': [27.0], 'dt': [0.09609870774790222],
# 'hidden_dim': [596.0], 'l2reg': [0.24927677946969878],
# 'layers': [4.0], 'lr': [0.005516656601005163],
# 'lr2': [1.897157209816416e-05], 'n_updates': [4.0]}

# %%
args = ObjectView(
    dict(
        dataset_size=200,
        fps=10,
        samples=100,
        num_epochs=80000,
        seed=0,
        loss="l1",
        act="softplus",
        hidden_dim=500,
        output_dim=1,
        layers=3,
        n_updates=1,  # 6,#4,
        lr=1e-3,  # 5.5e-3,
        lr2=2e-5,
        dt=0.1,
        model="gln",
        batch_size=68,
        l2reg=5.7e-7,
    )
)
rng = jax.random.PRNGKey(args.seed)

# %%


# %%

# %%
data = new_get_dataset(
    rng + 2,
    t_span=[0, args.dataset_size],
    fps=args.fps,
    samples=args.samples,
    test_split=0.9,
)

# %%
best_params = None
best_loss = np.inf


loss = make_loss(args)

# %%
opti = optimizers.adam

# %%
init_random_params, nn_forward_fn = extended_mlp(args)
_, init_params = init_random_params(rng + 1, (-1, 4))

HyperparameterSearch.nn_forward_fn = nn_forward_fn
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = opti(
    3e-4
)  ##lambda i: jnp.select([i<10000, i>= 10000], [args.lr, args.lr2]))
opt_state = opt_init(init_params)


train(args, model, data, rng)


@jax.jit
def update_derivative(
    i, opt_state, batch, l2reg, params
):  # iteration+offset, opt_state, batch, args.l2reg
    param_update = jax.grad(loss, 0)(params, batch, l2reg)
    new_state = opt_update(i, param_update, opt_state)
    leaves, _ = tree_flatten(get_params(new_state))
    infinities = sum((~jnp.isfinite(param)).sum() for param in leaves)

    def true_fun(x):
        # No introducing NaNs.
        return new_state, params

    def false_fun(x):
        # No introducing NaNs.
        return opt_state, params

    return jax.lax.cond(infinities == 0, 0, true_fun, 0, false_fun)


# %%
(nn_forward_fn, init_params) = model
data = {k: jax.device_put(v) for k, v in data.items()}


# %%
def make_new_params(params):
    rng = jax.random.PRNGKey(0)
    all_new_params = []
    for i in range(len(params)):
        new_params = []
        for j in range(len(params[i])):
            p = params[i][j]
            n_in = p.shape[0]
            n_out = 0 if len(p.shape) == 1 else p.shape[1]

            scaling = np.sqrt(6) / np.sqrt(n_in + n_out)
            new_p = jax.random.normal(rng, p.shape)

            if n_out > 0:
                if n_in >= n_out:
                    new_p = jnp.linalg.qr(new_p)[0]
                else:
                    new_p = jnp.linalg.qr(new_p.T)[0].T

            new_p *= scaling
            rng += 1

            new_params.append(new_p)
        new_params = tuple(new_params)
        all_new_params.append(new_params)
    return all_new_params


# %%
for _i in range(10000):
    print("Running", _i)
    print("Cur best", str(best_loss))

    best_small_loss = np.inf
    iteration = 0
    train_losses, test_losses = [], []

    lr = args.lr
    _, init_params = init_random_params(rng + 1, (-1, 4))
    rng += 1
    opt_init, opt_update, get_params = opti(lr)
    init_params = make_new_params(init_params)
    opt_state = opt_init(init_params)
    bad_iterations = 0
    offset = 0

    while iteration < 20000:
        iteration += 1
        rand_idx = jax.random.randint(rng, (args.batch_size,), 0, len(data["x"]))
        rng += 1

        batch = (data["x"][rand_idx], data["dx"][rand_idx])

        # Compute derivative at halfway point:
        half_state, params = update_derivative(
            iteration + offset, opt_state, batch, args.l2reg, get_params(opt_state)
        )
        half_params = get_params(half_state)
        opt_state, _ = update_derivative(
            iteration + offset, opt_state, batch, args.l2reg, half_params
        )
        params = get_params(opt_state)

        del half_params
        del half_state

        small_loss = loss(params, batch, 0.0)

        new_small_loss = False
        if small_loss < best_small_loss:
            best_small_loss = small_loss
            new_small_loss = True

        if (
            jnp.isnan(small_loss).sum()
            or new_small_loss
            or (iteration % 500 == 0)
            or (iteration < 1000 and iteration % 100 == 0)
        ):
            params = get_params(opt_state)
            train_loss = loss(params, (data["x"], data["dx"]), 0.0) / len(data["x"])
            train_losses.append(train_loss)
            test_loss = loss(params, (data["test_x"], data["test_dx"]), 0.0) / len(
                data["test_x"]
            )
            test_losses.append(test_loss)

            if iteration >= 1000 and test_loss > 2.1:
                # Only good seeds allowed!
                break

            if test_loss < best_loss:
                best_loss = test_loss
                best_params = copy(params)
                bad_iterations = 0
                offset += iteration
                iteration = 0  # Keep going since this one is so good!

            if jnp.isnan(test_loss).sum():
                break

            print(
                f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}"
            )

        bad_iterations += 1

    import pickle as pkl

    if best_loss < np.inf:
        pkl.dump(
            {"params": best_params, "args": args},
            open("params_for_loss_{}_nupdates=1.pkl".format(best_loss), "wb"),
        )



# %%
importlib.reload(lnn)

# %%

# %%
loss(best_params, (data["test_x"], data["test_dx"]), 0.0) / len(data["test_x"])

# %%

# %%

# %%
best_loss

# %%

# %%
