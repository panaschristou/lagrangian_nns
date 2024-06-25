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
import pickle as pkl
from copy import deepcopy as copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from celluloid import Camera
from IPython.display import HTML
from jax import jit
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.tree_util import tree_flatten
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from lnn.hyperopt import HyperparameterSearch
from lnn.hyperopt.HyperparameterSearch import (extended_mlp, make_loss,
                                               new_get_dataset, train)
from lnn.lnn import custom_init


# %%
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


# %%
jjit = jax.jit
# jjit = lambda _: _
ic = lambda *args, **kwargs: None
# from icecream import ic


# %%
# replace the lagrangian with a parameteric model
def learned_dynamics(params):
    @jjit
    def dynamics(q, q_t):
        #     assert q.shape == (2,)
        state = jnp.concatenate([q, q_t])
        out = nn_forward_fn(params, state)
        return jnp.squeeze(out, axis=-1)

    return dynamics


# %%
def raw_lagrangian_eom(lagrangian, state, t=None):
    # state is all q, then all q_t.
    vlagrangian = jjit(jax.vmap(lagrangian, (0, 0), 0))

    # Sum Lagrangians for all interacting coordinates.
    @jjit
    def lagrangian_fnc(q, q_t):
        # Assume q is [N,]
        # Move into [q_i-1, q_i, q_i+1]

        q_min = jnp.roll(q, shift=+1)
        q_plus = jnp.roll(q, shift=-1)

        q_t_min = jnp.roll(q_t, shift=+1)
        q_t_plus = jnp.roll(q_t, shift=-1)

        all_q = jnp.stack([q_min, q, q_plus], axis=-1)
        all_q_t = jnp.stack([q_t_min, q_t, q_t_plus], axis=-1)
        ic(all_q.shape, all_q_t.shape)
        return jnp.sum(vlagrangian(all_q, all_q_t))

    @jjit
    def conv_fnc(q, q_t):
        # Assume q is [N,]
        ic(q.shape, q_t.shape)
        q_tt = jnp.linalg.pinv(jax.hessian(lagrangian_fnc, 1)(q, q_t)) @ (
            jax.grad(lagrangian_fnc, 0)(q, q_t)
            - jax.jacobian(jax.jacobian(lagrangian_fnc, 1), 0)(q, q_t) @ q_t
        )
        ic(q.shape, q_tt.shape)
        return jnp.array([q_t, q_tt])

    @jjit
    def fnc(state):
        q, q_t = jnp.split(state, 2)

        out = conv_fnc(q, q_t)
        ic(out.shape, "first")
        out = jnp.concatenate([out[0], out[1]])
        ic(out.shape)
        return out

    return fnc(state)


# %% [markdown]
# ### Now, let's load the best model. To generate more models, see the code below.


# %%
# loaded = pkl.load(open('', 'rb'))

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
        "hidden_dim": 400,
        "input_dim": 6,  # adjacent points.
        "output_dim": 1,
        "layers": 3,
        "n_updates": 1,
        "lr": 0.001,
        "lr2": 2e-05,
        "dt": 0.1,
        "model": "gln",
        "batch_size": 512,
        "l2reg": 5.7e-07,
        "gridsize": 10,
    }
)
# args = loaded['args']
rng = jax.random.PRNGKey(args.seed)

# %%


# %%
dx = 0.1  # (lambda _x: _x[1]-_x[0])(jnp.linspace(0, 1, num=args.gridsize))


# %%
@jit
def analytic_fn(state):
    q, q_t = jnp.split(state, 2)

    q_plus = jnp.roll(q, shift=-1)
    q_min = jnp.roll(q, shift=+1)

    q_x = (q_plus - q_min) / (2 * dx)
    q_xx = (q_plus - 2 * q + q_min) / (2 * dx)

    # Wave equation with constraint:
    q_tt = q_xx
    return jnp.concatenate([q_t, q_tt])


# %%
q0 = jnp.exp(-((jnp.linspace(0, 1, num=args.gridsize) - 0.5) ** 2) / 2 / 0.05**2)
qt0 = jnp.zeros(args.gridsize)
state0 = jnp.concatenate([q0, qt0])

# %%
plt.plot(q0)
plt.plot(analytic_fn(state0)[args.gridsize :])

# %%
max_t = 30
all_t = jnp.linspace(0, max_t, num=100)

plot_gridsize = args.gridsize

q0 = jnp.exp(-((jnp.linspace(0, 1, num=plot_gridsize) - 0.5) ** 2) / 2 / 0.05**2)
qt0 = jnp.zeros(plot_gridsize)
state0 = jnp.concatenate([q0, qt0])


def ofunc(y, t=None):
    return analytic_fn(y)


state_t = odeint(ofunc, state0, all_t)


fig, ax = plt.subplots(1, 1)
cam = Camera(fig)

for i in range(len(state_t)):
    ax.plot((lambda _x: _x - jnp.average(_x))(state_t[i, :plot_gridsize]), c="k")
    cam.snap()

HTML(cam.animate().to_jshtml())

# %%

# %%
vfnc = jax.jit(jax.vmap(analytic_fn, 0, 0))

# %%
batch = 1
minibatch_per = 5000


@jjit
def get_derivative_dataset(rng):
    # randomly sample inputs
    num = int(args.gridsize / 10)

    unsmooth_y0 = jax.random.normal(rng, (batch * minibatch_per, args.gridsize))
    unsmooth_yt0 = jax.random.normal(rng + 1, (batch * minibatch_per, args.gridsize))

    y0 = jnp.concatenate(
        [
            sum(
                [
                    jnp.roll(unsmooth_y0, i, axis=1) * 2 ** (-((num / 2 - i) ** 2) / 10)
                    for i in range(num)
                ]
            ),
            sum(
                [
                    jnp.roll(unsmooth_yt0, i, axis=1)
                    * 2 ** (-((num / 2 - i) ** 2) / 10)
                    for i in range(num)
                ]
            ),
        ],
        axis=1,
    )

    return y0, vfnc(y0)


# %%
best_params = None
best_loss = np.inf

# %%

# %%
init_random_params, nn_forward_fn = extended_mlp(args)

HyperparameterSearch.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng + 1, (-1, args.input_dim))
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init([[l2 / 200.0 for l2 in l1] for l1 in init_params])


# train(args, model, data, rng);


@jjit
def loss(params, batch, l2reg):
    state, targets = batch  # _rk4
    #     leaves, _ = tree_flatten(params)
    #     l2_norm = sum(jnp.vdot(param, param) for param in leaves)
    preds = jax.vmap(partial(raw_lagrangian_eom, learned_dynamics(params)))(state)
    ic(preds.shape, targets.shape)
    # preds=(1000, 100, 2), targets=(1000, 200)
    return jnp.sum(jnp.abs(preds - targets))  # + l2reg*l2_norm/args.batch_size


@jjit
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
total_epochs = 30
train_losses, test_losses = [], []

# %%
nn_forward_fn(init_params, jnp.zeros((10, 6)))

# %%
lr = 1e-3  # 1e-3

final_div_factor = 1e4


# OneCycleLR:
@jjit
def OneCycleLR(pct):
    # Rush it:
    start = 0.3  # 0.2
    pct = pct * (1 - start) + start
    high, low = lr, lr / final_div_factor

    scale = 1.0 - (jnp.cos(2 * jnp.pi * pct) + 1) / 2

    return low + (high - low) * scale


opt_init, opt_update, get_params = optimizers.adam(OneCycleLR)

init_params = custom_init(init_params, seed=1)

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
batch_data = (
    get_derivative_dataset(rng)[0][:1000],
    get_derivative_dataset(rng)[1][:1000],
)
print(batch_data[0].shape)

# %%
plt.plot(batch_data[0][0])
plt.figure()
plt.plot(batch_data[1][0])

# %%
loss(init_params, [_x[:1] for _x in batch_data], 0.0) / len(batch_data[0])

# %%
update_derivative(0.0, opt_state, [_x[:1] for _x in batch_data], 0.0)


# %%
# best_loss = np.inf
# best_params = None
total_epochs = 100

# %%
rng += 3

# %%
if True:
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
            if batch_data[0].shape[0] == 0:
                break
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

# %%
opt_state = opt_init(best_params)

# %%
p = get_params(opt_state)

# %%
# pkl.dump(
#     best_params,
#     open('best_wave_equation_full_v1_{}.pt'.format(best_loss), 'wb')
# )

# %%
# best_params = pkl.load(open('best_wave_equation_v2_0.4340578317642212.pt', 'rb'))

# %%
# opt_state = opt_init(best_params)

# %% [markdown]
# ### Make sure the args are the same:

# %%
# opt_state = opt_init(loaded['params'])

# %%
rng + 7

# %%
max_t = 300
all_t = jnp.linspace(0, max_t, num=max_t * 5)

plot_gridsize = 100  # args.gridsize

q0 = jnp.exp(-((jnp.linspace(0, 1, num=plot_gridsize) - 0.5) ** 2) / 2 / (0.05) ** 2)
qt0 = jnp.zeros(plot_gridsize)
state0 = jnp.concatenate([q0, qt0])


def ofunc(y, t=None):
    return analytic_fn(y)


state_t = odeint(ofunc, state0, all_t)

# %%
t = state_t
tall = jax.device_get(t)
p = get_params(opt_state)

# %%
pred_tall = jax.device_get(
    odeint(partial(raw_lagrangian_eom, learned_dynamics(p)), state0, all_t)
)

# %%
pred_tall.shape, tall.shape


# %%
def energy(state):
    # state is [q, qt]
    q, q_t = jnp.split(state, 2)
    # Invariance:
    q_t -= jnp.average(q_t)
    q_plus = jnp.roll(q, shift=-1)
    q_min = jnp.roll(q, shift=+1)
    q_x = (q_plus - q_min) / (2 * 0.02040816)

    E = jnp.sum(0.5 * q_t**2 * 455.88 + 0.5 * q_x**2)
    return E


venergy = jax.vmap(energy, 0, 0)

# %%
true_en = venergy(tall)
pred_en = venergy(pred_tall)

plt.plot(true_en)

# %%
plt.rc("font", family="serif")


fig, allax = plt.subplots(2, 1, figsize=(4 * 3, 3 * 2))
# allax = allax.T
mmult = 1.5

# Swap zone:
cam = Camera(fig)
# for j, i in enumerate((np.r_[0, 100, 499]*mmult).astype(int)):
for i in tqdm(
    range(0, int(499 * mmult), 2)
):  # len(pred_tall), 1))):#tqdm(range(0, len(pred_tall), 1)):
    #     ax = allax[j]

    ax = allax
    cax = ax[0]
    cax.plot(
        np.arange(plot_gridsize),
        (lambda _: _ - np.average(_))(tall[i, :plot_gridsize]),
        color="b",
        linewidth=2,
        label="Truth",
    )
    cax.plot(
        np.arange(plot_gridsize),
        (lambda _: _ - np.average(_))(pred_tall[i, :plot_gridsize]),
        linestyle="--",
        color="orange",
        linewidth=2,
        label="Prediction",
    )
    cax.set_xlabel("Grid Index")
    cax.set_ylabel("Relative Amplitude")
    cax.set_ylim(-0.5, 1.0)

    cax = ax[1]
    (t1,) = cax.plot(
        np.arange(i + 1), true_en[: i + 1], color="b", linewidth=2, label="Truth"
    )
    (t2,) = cax.plot(
        np.arange(i + 1),
        pred_en[: i + 1],
        linestyle="--",
        color="orange",
        linewidth=2,
        label="Prediction",
    )
    cax.scatter(i, energy(pred_tall[i]), color="k")
    cax.set_xlabel("Time")
    cax.set_ylabel("Total Energy")
    cax.set_ylim(0, 250)
    cax.set_xlim(-20, 520 * mmult)
    cax.legend([t1, t2], ["Truth", "Prediction"], loc="lower left")

    plt.tight_layout()
    cam.snap()

ani = cam.animate()
HTML(ani.to_jshtml())

# plt.savefig('wave.pdf')

# %%
ani.save("wave_equation.mp4", fps=24)

# %%
ani.save("wave_equation.gif", writer="imagemagick", fps=24)


# %%

# %%

# %%


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
plt.rc("font", family="serif")

# %% [markdown]
# Let's compare energy for a variety of initial conditions:

# %%

# %%
all_errors = []
for i in tqdm(range(40)):
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
            partial(raw_lagrangian_eom, learned_dynamics(p)),
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

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Plots made down here:

# %%
total_predicted_energy_b = np.load("baseline_dblpend_energy.npy")
pred_tall_b = np.load("baseline_dblpend_prediction.npy")

# %%
tall = np.array(tall)
plt.plot(new_dataset["x"][:500, 0])
plt.plot(pred_tall[:500, 0])  # [:100, 0])
plt.ylabel(r"$\theta_1$")
plt.xlabel("Time")

# %%
jnp.max(jax.vmap(kinetic_energy, 0, 0)(new_dataset["x"][:]))

# %%
jnp.max(jnp.abs(jax.vmap(potential_energy, 0, 0)(new_dataset["x"][:])))

# %% [markdown]
# We set the scale of the system as the max potential energy of the double
# pendulum:
#
# $9.8\times1\times1 + 9.8\times1\times2=29.4$

# %%

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

plt.plot(jnp.abs(total_predicted_energy - total_true_energy) / scale)

plt.ylabel("Absolute Error in Total Energy/Max Potential Energy")
plt.xlabel("Time")
plt.ylim(-0.06, 0.01)

# %%

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
# np.save('lnn_dblpend_energy.npy', total_predicted_energy)
# np.save('lnn_dblpend_prediction.npy', pred_tall)

# %% [markdown]
# Let's compare:

# %%
tall = np.array(tall)
fig, ax = plt.subplots(2, 2, sharey=True)


for i in range(2):
    if i == 1:
        start = 1400
        end = 1500
    if i == 0:
        start = 0
        end = 100

    dom = np.linspace(start / 10, end / 10, num=end - start)
    ax[0, i].plot(dom, pred_tall[start:end, 0], label="LNN")  # [:100, 0])
    ax[0, i].plot(dom, pred_tall_b[start:end, 0], label="Baseline")  # [:100, 0])
    ax[0, i].plot(dom, new_dataset["x"][start:end, 0], label="Truth")
    # ax[0].set_xlabel('Time')
    ax[1, i].plot(
        dom, -new_dataset["x"][start:end, 0] + pred_tall[start:end, 0], label="LNN"
    )  # [:100, 0])
    ax[1, i].plot(
        dom,
        -new_dataset["x"][start:end, 0] + pred_tall_b[start:end, 0],
        label="Baseline",
    )  # [:100, 0])
    if i == 0:
        ax[0, i].set_ylabel(r"$\theta_1$")
        ax[1, i].set_ylabel(r"Error in $\theta_1$")

    ax[1, i].set_xlabel("Time")
    if i == 0:
        ax[0, i].legend()
        ax[1, i].legend()


for i in range(2):
    ax[i, 0].spines["right"].set_visible(False)
    ax[i, 1].spines["left"].set_visible(False)
    #     ax[i, 0].yaxis.tick_left()
    #     ax[i, 0].tick_params(labelright='off')
    ax[i, 1].yaxis.tick_right()

for i in range(2):
    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax[i, 0].transAxes, color="k", clip_on=False)
    ax[i, 0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax[i, 0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax[i, 1].transAxes)  # switch to the bottom axes
    ax[i, 1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax[i, 1].plot((-d, +d), (-d, +d), **kwargs)

plt.tight_layout()
plt.savefig("discrepancy_plot.pdf")

# %%
start = 0
end = 9999
dom = np.linspace(start / 10, end / 10, num=end - start)

scale = 29.4

fig, ax = plt.subplots(2, 1, sharex=True)
# translation = jnp.min(total_true_energy) + 1
# total_true_energy -= translation
# total_predicted_energy -= translation

ax[0].plot(dom, (total_predicted_energy), label="LNN")
ax[0].plot(dom, (total_predicted_energy_b), label="Baseline")
ax[0].plot(dom, (total_true_energy), label="Truth")
ax[0].set_ylabel("Total Energy")
ax[0].set_xlabel("Time")
ax[0].set_ylim(-15, 0)
ax[0].legend()

ax[1].plot(dom, (total_predicted_energy - total_true_energy) / scale, label="LNN")
ax[1].plot(
    dom, (total_predicted_energy_b - total_true_energy) / scale, label="Baseline"
)
ax[1].set_ylabel("Error in Total Energy\n/Max Potential Energy")
ax[1].set_xlabel("Time")
ax[1].set_ylim(-0.06, 0.01)
ax[1].legend()


plt.tight_layout()
plt.savefig("energy_discrepancy_plot.pdf")

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
    import HyperparameterSearch

    HyperparameterSearch.nn_forward_fn = nn_forward_fn
    _, init_params = init_random_params(rng + 1, (-1, 4))
    rng += 1
    model = (nn_forward_fn, init_params)
    opt_init, opt_update, get_params = optimizers.adam(
        3e-4
    )  ##lambda i: jnp.select([i<10000, i>= 10000], [args.lr, args.lr2]))
    opt_state = opt_init(init_params)
    loss = make_loss(args)
    from copy import deepcopy as copy

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
