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
from jax import jit
from jax.experimental.ode import odeint
from functools import partial
from matplotlib import pyplot as plt

from jax.example_libraries import optimizers
from lnn.hyperopt.HyperparameterSearch import learned_dynamics, extended_mlp
from lnn.hyperopt.HyperparameterSearch import new_get_dataset
from lnn.experiment_dblpend.data import get_trajectory_analytic
from lnn.experiment_dblpend.physics import analytical_fn
from lnn.hyperopt import HyperparameterSearch
from lnn.hyperopt.HyperparameterSearch import make_loss, train
from jax.tree_util import tree_flatten
from copy import deepcopy as copy
import pickle as pkl
from tqdm.notebook import tqdm


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

# %% [markdown]
# ### Now, let's load the best model. To generate more models, see the code below.

# %%
# loaded = pkl.load(open('./params_for_loss_0.29429444670677185_nupdates=1.pkl', 'rb'))

# %%
args = ObjectView(
    {
        "dataset_size": 200,
        "fps": 10,
        "samples": 100,
        "num_epochs": 80000,
        "seed": 0,
        "loss": "l1",
        "act": "relu_relu",
        "hidden_dim": 600,
        "output_dim": 2,
        "layers": 3,
        "n_updates": 1,
        "lr": 0.001,
        "lr2": 2e-05,
        "dt": 0.1,
        "model": "gln",
        "batch_size": 512,
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
minibatch_per = 2000
batch = 512


@jax.jit
def get_derivative_dataset(rng):
    # randomly sample inputs

    y0 = jnp.concatenate(
        [
            jax.random.uniform(rng, (batch * minibatch_per, 2)) * 2.0 * np.pi,
            (jax.random.uniform(rng + 1, (batch * minibatch_per, 2)) - 0.5) * 10 * 2,
        ],
        axis=1,
    )

    return y0, vfnc(y0)


# %%
best_params = None
best_loss = np.inf


# %%

# %%
# @jax.jit
def baseline_eom(baseline, state, t=None):
    q, q_t = jnp.split(state, 2)
    q = q % (2 * jnp.pi)
    q_tt = baseline(q, q_t)
    return jnp.concatenate([q_t, q_tt])


# %%
init_random_params, nn_forward_fn = extended_mlp(args)


HyperparameterSearch.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng + 1, (-1, 4))
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init([[l2 / 200.0 for l2 in l1] for l1 in init_params])

# train(args, model, data, rng);


@jax.jit
def loss(params, batch, l2reg):
    state, targets = batch  # _rk4
    leaves, _ = tree_flatten(params)
    l2_norm = sum(jnp.vdot(param, param) for param in leaves)
    preds = jax.vmap(partial(baseline_eom, learned_dynamics(params)))(state)
    return jnp.sum(jnp.abs(preds - targets)) + l2reg * l2_norm / args.batch_size


# @jax.jit
# def normalize_param_update(param_update):
#     new_params = []
#     num_weights = args.hidden_dim**2*3
#     gradient_norm = sum([jnp.sum(l2**2)
#                          for l1 in param_update
#                          for l2 in l1
#                          if len(l1) != 0])/num_weights
# #     gradient_norm = 1 +
#     for l1 in param_update:
#         if (len(l1)) == 0: new_params.append(()); continue
#         new_l1 = []
#         for l2 in l1:
#             new_l1.append(
#                 l2/gradient_norm
#             )

#         new_params.append(new_l1)

#     return new_params


@jax.jit
def update_derivative(i, opt_state, batch, l2reg):
    params = get_params(opt_state)
    param_update = jax.grad(lambda *args: loss(*args) / len(batch), 0)(
        params, batch, l2reg
    )
    #     param_update = normalize_param_update(param_update)
    params = get_params(opt_state)
    return opt_update(i, param_update, opt_state), params


best_small_loss = np.inf
(nn_forward_fn, init_params) = model
iteration = 0
total_epochs = 300
minibatch_per = 2000
train_losses, test_losses = [], []

lr = 4e-5  # 1e-3

final_div_factor = 1e4


# OneCycleLR:
@jax.jit
def OneCycleLR(pct):
    # Rush it:
    start = 0.2  # 0.2
    pct = pct * (1 - start) + start
    high, low = lr, lr / final_div_factor

    scale = 1.0 - (jnp.cos(2 * jnp.pi * pct) + 1) / 2

    return low + (high - low) * scale


opt_init, opt_update, get_params = optimizers.adam(OneCycleLR)

# init_params = custom_init(init_params, seed=0)
# init_params =
opt_state = opt_init(init_params)
# opt_state = opt_init(best_params)
bad_iterations = 0
print(lr)

# %% [markdown]
# Idea: add identity before inverse:

# %% [markdown]
# # Let's train it:

# %%
rng = jax.random.PRNGKey(0)

# %%
epoch = 0

# %%
batch_data = get_derivative_dataset(rng)[0][:10], get_derivative_dataset(rng)[1][:10]
print(batch_data[0].shape)

# %%
loss(get_params(opt_state), batch_data, 0.0)

# %%
opt_state, params = update_derivative(0.0, opt_state, batch_data, 0.0)

# %%
# best_loss = np.inf
# best_params = None

# %%
for epoch in tqdm(range(epoch, total_epochs)):
    epoch_loss = 0.0
    num_samples = 0
    all_batch_data = get_derivative_dataset(rng)
    for minibatch in range(minibatch_per):
        fraction = (epoch + minibatch / minibatch_per) / total_epochs
        batch_data = (
            all_batch_data[0][minibatch * batch : (minibatch + 1) * batch],
            all_batch_data[1][minibatch * batch : (minibatch + 1) * batch],
        )
        rng += 10
        opt_state, params = update_derivative(fraction, opt_state, batch_data, 1e-6)
        cur_loss = loss(params, batch_data, 0.0)
        epoch_loss += cur_loss
        num_samples += batch
    closs = epoch_loss / num_samples
    print("epoch={} lr={} loss={}".format(epoch, OneCycleLR(fraction), closs))
    if closs < best_loss:
        best_loss = closs
        best_params = [
            [copy(jax.device_get(l2)) for l2 in l1] if len(l1) > 0 else ()
            for l1 in params
        ]

# %% [markdown]
# Look at distribution of weights to make a better model?

# %%
best_params = pkl.load(open("best_dblpendulum_baseline_v5_900epoch.pt", "rb"))

# %%
# p = get_params(opt_state)

# %%
opt_state = opt_init(best_params)

# %%
# pkl.dump(
#     best_params,
#     open('best_dblpendulum_baseline_v5_900epoch.pt', 'wb')
# )

# %% [markdown]
# ### Make sure the args are the same:

# %%
# opt_state = opt_init(loaded['params'])

# %%
rng + 7

# %% [markdown]
# The seed: [8, 8] looks pretty good! Set args.n_updates=3, and the file params_for_loss_0.29429444670677185_nupdates=1.pkl.

# %%
plt.rc("font", family="serif")

# %%
max_t = 10
new_dataset = new_get_dataset(
    jax.random.PRNGKey(2),
    t_span=[0, max_t],
    fps=10,
    test_split=1.0,
    unlimited_steps=False,
)

# %%
t = new_dataset["x"][0, :]
tall = [jax.device_get(t)]
p = get_params(opt_state)

# %%
new_dataset["x"].shape

# %%
pred_tall = jax.device_get(
    odeint(
        partial(baseline_eom, learned_dynamics(p)),
        t,
        np.linspace(0, max_t, num=new_dataset["x"].shape[0]),
    )
)


# %%
@jit
def kinetic_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = jnp.split(state, 2)
    (t1, t2), (w1, w2) = q, q_dot

    T1 = 0.5 * m1 * (l1 * w1) ** 2
    T2 = (
        0.5
        * m2
        * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    )
    T = T1 + T2
    return T


@jit
def potential_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = jnp.split(state, 2)
    (t1, t2), (w1, w2) = q, q_dot

    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2
    return V


# %%
tall = np.array(tall)
plt.plot(new_dataset["x"][:30, 0])  # [:100, 0])
plt.plot(pred_tall[:30, 0])

plt.ylabel(r"$\theta_1$")
plt.xlabel("Time")

# %%

# %%
all_errors = []
for i in tqdm(range(100)):
    max_t = 100
    new_dataset = new_get_dataset(
        jax.random.PRNGKey(i),
        t_span=[0, max_t],
        fps=10,
        test_split=1.0,
        unlimited_steps=False,
    )
    t = new_dataset["x"][0, :]
    tall = [jax.device_get(t)]
    p = best_params
    pred_tall = jax.device_get(
        odeint(
            partial(baseline_eom, learned_dynamics(p)),
            t,
            np.linspace(0, max_t, num=new_dataset["x"].shape[0]),
        )
    )

    total_true_energy = jax.vmap(kinetic_energy, 0, 0)(new_dataset["x"][:]) + jax.vmap(
        potential_energy, 0, 0
    )(new_dataset["x"][:])
    total_predicted_energy = jax.vmap(kinetic_energy, 0, 0)(pred_tall[:]) + jax.vmap(
        potential_energy, 0, 0
    )(pred_tall[:])

    scale = 29.4

    # translation = jnp.min(total_true_energy) + 1
    # total_true_energy -= translation
    # total_predicted_energy -= translation

    cur_error = jnp.abs((total_predicted_energy - total_true_energy)[-1]) / scale
    all_errors.append(cur_error)

    print(i, "current error", jnp.average(all_errors))

# %% [markdown]
# # Can't run the 44th system. Just freezes...

# %%

# %%

# %% [markdown]
# ## Make plots:

# %%
total_true_energy = jax.vmap(kinetic_energy, 0, 0)(new_dataset["x"][:]) + jax.vmap(
    potential_energy, 0, 0
)(new_dataset["x"][:])
total_predicted_energy = jax.vmap(kinetic_energy, 0, 0)(pred_tall[:]) + jax.vmap(
    potential_energy, 0, 0
)(pred_tall[:])
scale = 29.4

# translation = jnp.min(total_true_energy) + 1
# total_true_energy -= translation
# total_predicted_energy -= translation

plt.plot((total_predicted_energy - total_true_energy) / scale)

plt.ylabel("Absolute Error in Total Energy/Max Potential Energy")
plt.xlabel("Time")
plt.ylim(-0.06, 0.01)

# %%
# np.save('baseline_dblpend_energy.npy', total_predicted_energy)
# np.save('baseline_dblpend_prediction.npy', pred_tall)

# %%
rng = jax.random.PRNGKey(int(1e9))

# %%
batch_data = (
    get_derivative_dataset(rng)[0][:100000],
    get_derivative_dataset(rng)[1][:100000],
)
print(batch_data[0].shape)

# %%
loss(best_params, batch_data, 0.0) / len(batch_data[0])

# %%

# %%

# %%

# %%

# %%
best_loss = np.inf
best_params = None

# %%
for _i in range(1000):
    print("Running", _i)
    print("Cur best", str(best_loss))

    init_random_params, nn_forward_fn = extended_mlp(args)

    HyperparameterSearch.nn_forward_fn = nn_forward_fn
    _, init_params = init_random_params(rng + 1, (-1, 4))
    rng += 1
    model = (nn_forward_fn, init_params)
    opt_init, opt_update, get_params = optimizers.adam(
        3e-4
    )  ##lambda i: jnp.select([i<10000, i>= 10000], [args.lr, args.lr2]))
    opt_state = opt_init(init_params)
    loss = make_loss(args)

    train(args, model, data, rng)

    @jax.jit
    def update_derivative(i, opt_state, batch, l2reg):
        params = get_params(opt_state)
        param_update = jax.grad(loss, 0)(params, batch, l2reg)
        leaves, _ = tree_flatten(param_update)
        infinities = sum((~jnp.isfinite(param)).sum() for param in leaves)

        def true_fun(x):
            # No introducing NaNs.
            return opt_update(i, param_update, opt_state), params

        def false_fun(x):
            # No introducing NaNs.
            return opt_state, params

        return jax.lax.cond(infinities == 0, 0, true_fun, 0, false_fun)

    best_small_loss = np.inf
    (nn_forward_fn, init_params) = model
    data = {k: jax.device_put(v) for k, v in data.items()}
    iteration = 0
    train_losses, test_losses = [], []
    lr = args.lr
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)
    bad_iterations = 0
    offset = 0

    while iteration < 20000:
        iteration += 1
        rand_idx = jax.random.randint(rng, (args.batch_size,), 0, len(data["x"]))
        rng += 1

        batch = (data["x"][rand_idx], data["dx"][rand_idx])
        opt_state, params = update_derivative(
            iteration + offset, opt_state, batch, args.l2reg
        )
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

            if iteration >= 1000 and test_loss > 1.5:
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
                lr = lr / 2
                opt_init, opt_update, get_params = optimizers.adam(lr)
                opt_state = opt_init(best_params)
                bad_iterations = 0

            print(
                f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}"
            )

        bad_iterations += 1

    if best_loss < np.inf:
        pkl.dump(
            {"params": best_params, "args": args},
            open("params_for_loss_{}_nupdates=1.pkl".format(best_loss), "wb"),
        )

# %%

# %%
loss(best_params, (data["test_x"], data["test_dx"]), 0.0) / len(data["test_x"])

# %%

# %%

# %%
best_loss

# %%

# %%
