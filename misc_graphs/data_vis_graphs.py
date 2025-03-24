import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gym

np.random.seed(1017)

### Helper Functions 

initial_point = np.array([10.61098536,  5.87720862, 34.48052002])
params = [10, 28, 8/3]
dt = 0.001

def dpdt(point, params=params): # position
    x, y, z = point
    sig, rho, beta = params
    
    new_x = y*dt*sig + x*(1 - dt*sig)
    new_y = x*dt*(rho - z) + y*(1 - dt)
    new_z = x*y*dt + z*(1 - dt*beta)
    return np.array([new_x, new_y, new_z])

def make_lorenz_rollout(num_samples):
    positions = [initial_point]
    for _ in range(num_samples):
        positions.append(dpdt(positions[-1]))
    positions = np.stack(positions)
    # Euclidean distance from t to t+1
    labels = np.sum(np.sqrt(np.square(positions[1:] - positions[:-1])), axis=1)
    return positions, labels

### Create the Gym environment and collect samples

num_rollouts = 100
env = gym.make("Pendulum-v1")

old_obs = []
rewards = []
for _ in range(num_rollouts):
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs = observation
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)

        old_obs.append(obs)
        rewards.append(reward)

    env.close()

old_obs = np.vstack(old_obs)
rewards = np.vstack(rewards)

# Lorenz Samples
l_5k, l_5k_labels   = make_lorenz_rollout(5_000)
l_20k, l_20k_labels = make_lorenz_rollout(20_000)
l_100k, l_100k_labels = make_lorenz_rollout(100_000)

# Pendulum Samples
pen_10k_idx = np.random.choice(len(old_obs), size=10_000, replace=False)
pen_5k_idx  = np.random.choice(len(old_obs), size=5_000,  replace=False)

pen_5k   = old_obs[pen_5k_idx]
pen_5k_l = rewards[pen_5k_idx]
pen_10k  = old_obs[pen_10k_idx]
pen_10k_l= rewards[pen_10k_idx]

### Build a 3x3 grid of subplots (all are 3D scenes)

fig = make_subplots(
    rows=3, cols=3,
    specs=[
        [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]
    ],
    subplot_titles=[
        "Lorenz 5k",   "Lorenz 20k",    "Lorenz 100k",
        "Pendulum 5k", "Pendulum 10k",  "Pendulum 20k",
        "PO-Pendulum 5k",   "PO-Pendulum 10k",    "PO-Pendulum 20k"
    ]
)

### Row 1: Lorenz (3D)
fig.add_trace(
    go.Scatter3d(
        x=l_5k[1:, 0], y=l_5k[1:, 1], z=l_5k[1:, 2],
        mode='markers',
        marker=dict(size=2, color=l_5k_labels, colorscale='Viridis')
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=l_20k[1:, 0], y=l_20k[1:, 1], z=l_20k[1:, 2],
        mode='markers',
        marker=dict(size=2, color=l_20k_labels, colorscale='Viridis')
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=l_100k[1:, 0], y=l_100k[1:, 1], z=l_100k[1:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=l_100k_labels,
            colorscale='Viridis',
            colorbar=dict(
                x=1,  # shift colorbar horizontally
                y=0.9,
                len=0.4,
                thickness=20,
                title='Euclidean Distance t->t+1'
            )
        )
    ),
    row=1, col=3
)

### Row 2: Pendulum (3D)
fig.add_trace(
    go.Scatter3d(
        x=pen_5k[:, 0], y=pen_5k[:, 1], z=pen_5k[:, 2],
        mode='markers',
        marker=dict(size=2, color=pen_5k_l[:, 0], colorscale='Cividis')
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=pen_10k[:, 0], y=pen_10k[:, 1], z=pen_10k[:, 2],
        mode='markers',
        marker=dict(size=2, color=pen_10k_l[:, 0], colorscale='Cividis')
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=old_obs[:, 0], y=old_obs[:, 1], z=old_obs[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=rewards[:, 0],
            colorscale='Cividis',
            colorbar=dict(
                x=1.0,
                y=0.3,
                len=0.75,
                thickness=20,
                title='Reward'
            )
        )
    ),
    row=2, col=3
)

### Row 3: Pendulum, but with z=0 (still a 3D scatter)
fig.add_trace(
    go.Scatter3d(
        x=pen_5k[:, 0],
        y=pen_5k[:, 1],
        z=np.zeros(pen_5k.shape[0]),
        mode='markers',
        marker=dict(size=2, color=pen_5k_l[:, 0], colorscale='Cividis')
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=pen_10k[:, 0],
        y=pen_10k[:, 1],
        z=np.zeros(pen_10k.shape[0]),
        mode='markers',
        marker=dict(size=2, color=pen_10k_l[:, 0], colorscale='Cividis')
    ),
    row=3, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=old_obs[:, 0],
        y=old_obs[:, 1],
        z=np.zeros(old_obs.shape[0]),
        mode='markers',
        marker=dict(size=2, color=rewards[:, 0], colorscale='Cividis')
    ),
    row=3, col=3
)

fig.update_layout(
    height=1200,
    width=1600,
    showlegend=False,
    title_text="Visualization of Datapoints"
)

fig.write_html("Vis_Graphs.html")
