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
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.experimental.ode import odeint
from functools import partial

from jax.example_libraries import optimizers


from lnn.hyperopt.HyperparameterSearch import extended_mlp


# %%

# %%

# %%
def lagrangian_eom(lagrangian, state, conditionals, t=None):
    q, q_t = jnp.split(state, 2)
    q = q / 10.0  # Normalize
    conditionals = conditionals / 10.0
    q_t = q_t
    q_tt = jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t, conditionals)) @ (
        jax.grad(lagrangian, 0)(q, q_t, conditionals)
        - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t, conditionals) @ q_t
    )
    return jnp.concatenate([q_t, q_tt])


# replace the lagrangian with a parameteric model
def learned_dynamics(params, nn_forward_fn):
    @jit
    def dynamics(q, q_t, conditionals):
        #     assert q.shape == (2,)
        state = jnp.concatenate([q, q_t, conditionals])
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)

    return dynamics


# %%
class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


# %%
@jax.jit
def qdotdot(q, q_t, conditionals):
    g = conditionals

    q_tt = g * (1 - q_t**2) ** (5.0 / 2) / (1 + 2 * q_t**2)

    return q_t, q_tt


@jax.jit
def ofunc(y, t=None):
    q = y[::3]
    q_t = y[1::3]
    g = y[2::3]

    q_t, q_tt = qdotdot(q, q_t, g)
    return jnp.stack([q_t, q_tt, jnp.zeros_like(g)]).T.ravel()


# %%
(jnp.tanh(jax.random.uniform(jax.random.PRNGKey(1), (1000,)) * 10 - 5) * 0.99999).max()

# %%
from matplotlib import pyplot as plt

# %%
plt.hist((jnp.tanh(jax.random.normal(jax.random.PRNGKey(1), (100,)) * 2) * 0.99999))


# %%
@partial(jax.jit, static_argnums=(1, 2), backend="cpu")
def gen_data(seed, batch, num):
    rng = jax.random.PRNGKey(seed)
    q0 = jax.random.uniform(rng, (batch,), minval=-10, maxval=10)
    qt0 = jax.random.uniform(rng + 1, (batch,), minval=-0.99, maxval=0.99)
    g = jax.random.normal(rng + 2, (batch,)) * 10

    y0 = jnp.stack([q0, qt0, g]).T.ravel()

    yt = odeint(
        ofunc,
        y0,
        jnp.linspace(0, 1, num=num),
    )

    qall = yt[:, ::3]
    qtall = yt[:, 1::3]
    gall = yt[:, 2::3]

    return (
        jnp.stack([qall, qtall]).reshape(2, -1).T,
        gall.reshape(1, -1).T,
        qdotdot(qall, qtall, gall)[1].reshape(1, -1).T,
    )


@partial(jax.jit, static_argnums=(1,))
def gen_data_batch(seed, batch):
    rng = jax.random.PRNGKey(seed)
    q0 = jax.random.uniform(rng, (batch,), minval=-10, maxval=10)
    qt0 = (
        jnp.tanh(jax.random.normal(jax.random.PRNGKey(1), (batch,)) * 2) * 0.99999
    )  # jax.random.uniform(rng+1, (batch,), minval=-1, maxval=1)
    g = jax.random.normal(rng + 2, (batch,)) * 10

    return (
        jnp.stack([q0, qt0]).reshape(2, -1).T,
        g.reshape(1, -1).T,
        qdotdot(q0, qt0, g)[1].reshape(1, -1).T,
    )


# %%
print(
    "qt",
    gen_data_batch(0, 128)[0][:5, 1],
    "g",
    gen_data_batch(0, 128)[1][:5, 0],
    "qtt",
    gen_data_batch(0, 128)[2][:5, 0],
)

# %%
from matplotlib import pyplot as plt

# %%
# qdotdot(jnp.array([0]), jnp.array([0.9]), jnp.array([10]))

# %%
# 0.29830917716026306 {'act': [4],
# 'batch_size': [27.0], 'dt': [0.09609870774790222],
# 'hidden_dim': [596.0], 'l2reg': [0.24927677946969878],
# 'layers': [4.0], 'lr': [0.005516656601005163],
# 'lr2': [1.897157209816416e-05], 'n_updates': [4.0]}

# %%
import pickle as pkl

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
        "act": "softplus",
        "hidden_dim": 500,
        "output_dim": 1,
        "layers": 4,
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
best_params = None
best_loss = np.inf

# %%

# %%
init_random_params, nn_forward_fn = extended_mlp(args)
rng = jax.random.PRNGKey(0)
_, init_params = init_random_params(rng, (-1, 3))
rng += 1

# %% [markdown]
# This is the output. Now, let's train it.

# %% [markdown]
# Idea: add identity before inverse:

# %% [markdown]
# # Let's train it:

# %%
best_small_loss = np.inf
iteration = 0
total_epochs = 100
minibatch_per = 3000
train_losses, test_losses = [], []

lr = 1e-3  # 1e-3

final_div_factor = 1e4


# OneCycleLR:
@jax.jit
def OneCycleLR(pct):
    # Rush it:
    start = 0.3  # 0.2
    pct = pct * (1 - start) + start
    high, low = lr, lr / final_div_factor

    scale = 1.0 - (jnp.cos(2 * jnp.pi * pct) + 1) / 2

    return low + (high - low) * scale


opt_init, opt_update, get_params = optimizers.adam(OneCycleLR)
from lnn import custom_init

init_params = custom_init(init_params, seed=0)
opt_state = opt_init(init_params)
# opt_state = opt_init(best_params)

# %%
plt.plot(OneCycleLR(jnp.linspace(0, 1, num=200)))
plt.yscale("log")
plt.title("lr schedule")


# %%
@jax.jit
def loss(params, cstate, cconditionals, ctarget):
    runner = jax.vmap(
        partial(lagrangian_eom, learned_dynamics(params, nn_forward_fn)), (0, 0), 0
    )
    preds = runner(cstate, cconditionals)[:, [1]]

    error = jnp.abs(preds - ctarget)
    # Weight additionally by proximity to c!
    error_weights = 1 + 1 / jnp.sqrt(1.0 - cstate[:, [1]] ** 2)

    return jnp.sum(error * error_weights) * len(preds) / jnp.sum(error_weights)


@jax.jit
def update_derivative(i, opt_state, cstate, cconditionals, ctarget):
    params = get_params(opt_state)
    param_update = jax.grad(lambda *args: loss(*args) / len(cstate), 0)(
        params, cstate, cconditionals, ctarget
    )
    params = get_params(opt_state)
    return opt_update(i, param_update, opt_state), params


# %%
epoch = 0

# %%
cstate, cconditionals, ctarget = gen_data_batch(epoch, 128)

# %%
loss(get_params(opt_state), cstate, cconditionals, ctarget)

# %%
update_derivative(0, opt_state, cstate, cconditionals, ctarget);

# %%
rng = jax.random.PRNGKey(0)

# %%
epoch = 0

# %%
from tqdm.notebook import tqdm

# %%
gen_data_batch(0, 128)[0].shape

# %%
cconditionals[:5]

# %%
cstate[:5]

# %%
ctarget[:5]

# %%
best_loss = np.inf
best_params = None

# %%
from copy import deepcopy as copy

# %%
for epoch in tqdm(range(epoch, total_epochs)):
    epoch_loss = 0.0
    num_samples = 0
    batch = 512
    ocstate, occonditionals, octarget = gen_data_batch(epoch, minibatch_per * batch)
    for minibatch in range(minibatch_per):
        fraction = (epoch + minibatch / minibatch_per) / total_epochs
        s = np.s_[minibatch * batch : (minibatch + 1) * batch]

        cstate, cconditionals, ctarget = ocstate[s], occonditionals[s], octarget[s]
        opt_state, params = update_derivative(
            fraction, opt_state, cstate, cconditionals, ctarget
        )
        rng += 10

        cur_loss = loss(params, cstate, cconditionals, ctarget)

        epoch_loss += cur_loss
        num_samples += len(cstate)
    closs = epoch_loss / num_samples
    print("epoch={} lr={} loss={}".format(epoch, OneCycleLR(fraction), closs))
    if closs < best_loss:
        best_loss = closs
        best_params = [
            [copy(jax.device_get(l2)) for l2 in l1] if len(l1) > 0 else ()
            for l1 in params
        ]

# %%

# %%
# pkl.dump({'params': best_params, 'description': 'q and g are divided by 10. hidden=500. act=Softplus'},
#          open('best_sr_params_v2.pkl', 'wb'))

# %%
best_params = pkl.load(open("best_sr_params_v2.pkl", "rb"))["params"]

# %%
opt_state = opt_init(best_params)

# %%
cstate, cconditionals, ctarget = gen_data(0, 1, 50)

# %%
cstate.shape

# %%
plt.plot(cstate[:, 1])

# %%
params = get_params(opt_state)

# %%

# %%
plt.rc("font", family="serif")

# %%
fig, ax = plt.subplots(1, 1, figsize=(4 * 1, 4 * 1), sharex=True, sharey=True)
ax_idx = [(i, j) for i in range(1) for j in range(1)]

for i in tqdm(range(1)):
    ci = ax_idx[i]

    cstate, cconditionals, ctarget = gen_data((i + 4) * (i + 1), 1, 50)

    runner = jax.jit(
        jax.vmap(
            partial(lagrangian_eom, learned_dynamics(params, nn_forward_fn)), (0, 0), 0
        )
    )

    @jax.jit
    def odefunc_learned(y, t):
        return jnp.concatenate((runner(y[None, :2], y[None, [2]])[0], jnp.zeros(1)))

    yt_learned = odeint(
        odefunc_learned,
        jnp.concatenate([cstate[0], cconditionals[0]]),
        np.linspace(0, 1, 50),
    )

    cax = ax  # [ci[0], ci[1]]
    cax.plot(cstate[:, 1], label="Truth")
    cax.plot(yt_learned[:, 1], label="Learned")
    cax.legend()
    if ci[1] == 0:
        cax.set_ylabel("Velocity of particle/Speed of light")
    if ci[0] == 0:
        cax.set_xlabel("Time")

    cax.set_ylim(-1, 1)

plt.title("Lagrangian NN - Special Relativity")
plt.tight_layout()
plt.savefig("sr_lnn.png", dpi=150)

# %%
