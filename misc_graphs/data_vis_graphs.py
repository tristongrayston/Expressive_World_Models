import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import gym

# doesn't help that much because the gyms random actions will be random anyway, but wtvr.

np.random.seed(1017)

### Helper Functions 

initial_point = np.array([10.61098536,  5.87720862, 34.48052002])
params = [10, 28, 8/3]
dt = 0.001

def dpdt(point, params=params): #position

    x,y,z = point
    sig, rho, beta = params
    
    new_x = y*dt*sig + x*(1-dt*sig)
    new_y = x*dt*(rho-z) + y*(1-dt)
    new_z = x*y*dt + z*(1-dt*beta)
    return np.array([new_x, new_y, new_z])

num_rollouts = 100 # 100 * 200 = 20,000 training samples. should be fine..?
env = gym.make("Pendulum-v1")

def make_lorenz_rollout(num_samples):
    positions = []
    positions.append(initial_point)

    # your dataset
    for _ in range(num_samples):
        positions.append(dpdt(positions[-1]))

    positions = np.stack(positions)

    labels = np.sum(np.sqrt(np.square(positions[1:num_samples+1] - positions[:num_samples])), axis=1)

    return positions, labels

old_obs = []
rewards = []
for _ in range(num_rollouts):

    observation, _ = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs = observation

        action = env.action_space.sample()

        # Take a step in the environment
        observation, reward, terminated, truncated, _ = env.step(action)

        old_obs.append(obs)
        rewards.append(reward)

    # Once the episode is done, close the environment
    env.close()

old_obs = np.vstack(old_obs)
rewards = np.vstack(rewards)

pen_max_sams = num_rollouts * 100

n_samples = 5_000

pen_five_k = np.random.randint(0, pen_max_sams, size=5_000)
pen_ten_k = np.random.randint(0, pen_max_sams, size=10_000)

# Lorenz Samples

l_5k, l_5k_labels = make_lorenz_rollout(5_000)
l_20k, l_20k_labels = make_lorenz_rollout(20_000)
l_100k, l_100k_labels = make_lorenz_rollout(100_000)

# Gym Samples

pen_10k_idx = np.random.choice(20000, size=10000, replace=False)
pen_5k_idx = np.random.choice(20000, size=5000, replace=False)


pen_10k = old_obs[pen_10k_idx]
pen_10k_l = rewards[pen_10k_idx]

pen_5k = old_obs[pen_5k_idx]
pen_5k_l = rewards[pen_5k_idx]


### Plotly Plots

fig = make_subplots(
    rows=2, cols=3,
    specs=[
        [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
    ],
    subplot_titles=[
        "Lorenz 5k Samples",
        "Lorenz 20k Samples",
        "Lorenz 100k Samples",
        "Pendulum 5k Samples",
        "Pendulum 10k Samples",
        "Pendulum 20k Samples"
    ]
)

# For Lorenz plots, we use positions from index 1 onward (so that labels and positions match)
fig.add_trace(
    go.Scatter3d(
        x=l_5k[1:, 0],
        y=l_5k[1:, 1],
        z=l_5k[1:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=l_5k_labels,
            colorscale='Viridis'
        )
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=l_20k[1:, 0],
        y=l_20k[1:, 1],
        z=l_20k[1:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=l_20k_labels,
            colorscale='Viridis'
        )
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=l_100k[1:, 0],
        y=l_100k[1:, 1],
        z=l_100k[1:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=l_100k_labels,
            colorscale='Viridis',
            colorbar=dict(
                x=1.2,
                y=0.8,
                len=0.4,
                thickness=20,
                title='Euclidean Distance t->t+1'
            )
        )
    ),
    row=1, col=3
)


# Gym 5k Samples
fig.add_trace(
    go.Scatter3d(
        x=pen_5k[:, 0],
        y=pen_5k[:, 1],
        z=pen_5k[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pen_5k_l[:, 0],
            colorscale='Cividis'
        )
    ),
    row=2, col=1
)

# Gym 10k Samples
fig.add_trace(
    go.Scatter3d(
        x=pen_10k[:, 0],
        y=pen_10k[:, 1],
        z=pen_10k[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pen_10k_l[:, 0],
            colorscale='Cividis'
        )
    ),
    row=2, col=2
)

# Gym Full Samples (all gym data)
fig.add_trace(
    go.Scatter3d(
        x=old_obs[:, 0],
        y=old_obs[:, 1],
        z=old_obs[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=rewards[:, 0],
            colorscale='Cividis',
            colorbar=dict(
                x=1.2,        # shift colorbar horizontally
                y=0.2,
                len=0.4,       # how tall the bar is
                thickness=20,  # how wide
                title='Reward'
            )
        )
    ),
    row=2, col=3
)

fig.update(layout_showlegend=False)
fig.update_layout(height=1200, width=1600, title_text="Visualization of Datapoints from 2 Environments")
fig.write_html("Vis_Graphs.html")
