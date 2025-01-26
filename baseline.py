import gym
import numpy as np
import torch as t
from ppo import PPO  # Make sure 'ppo.py' is in the same folder or in your Python path
import matplotlib.pyplot as plt

device = t.device("cuda" if t.cuda.is_available() else "cpu")
def train_ppo(env_name):
    env = gym.make(env_name)
    actionspace = env.action_space.shape[0]
    obsspace = env.observation_space.shape[0]
    #print(f"actionspace: {actionspace}, obs space {obsspace}")
    agent = PPO(
        ob_space=obsspace,
        actions=actionspace,
        n_batches=10,
        gamma=0.99,
        lam=0.95,
        kl_coeff=0.2,
        clip_rewards=False,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0,
        a_lr=1e-4,
        c_lr=1e-4,
        device='cpu',
        max_ts=100,

        # Any custom kwargs can also be passed in here. For example:
        rollouts_per_batch=1,
        max_timesteps_per_episode=1000,
        n_updates_per_iteration=3,
    )
    
    # 3. Train the agent
    total_timesteps = 1_000_000  # Decide how long you want to train
    agent.learn(total_timesteps=total_timesteps, env=env)
    
    # 4. (Optional) Close the environment
    env.close()
    return agent

def test_ppo(ppo_agent, env_name):
    # Create the environment in 'human' render mode so it shows visualization
    env = gym.make(env_name, render_mode='human')

    # Reset the environment to get the initial observation
    observation, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Get the action from the trained PPO agent
        vect_obs = t.tensor(observation, dtype=t.float32, device='cpu')
        action, _ = ppo_agent.get_action(vect_obs)  # or .predict(...)

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

    # Once the episode is done, close the environment
    env.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_eps_rewards(agent, window_size=10):
    """
    Plots episode rewards over time and includes a moving average trend line.
    
    Parameters:
    -----------
    agent : object
        Your agent object, which should have a logger dict containing 'eps_rewards'.
    window_size : int
        The size of the window over which to compute the moving average.
    """

    # Extract the list of episode rewards
    rewards = agent.logger['eps_rewards']

    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot the raw episode rewards
    plt.plot(rewards, marker='o', linestyle='-', color='b', label='Episode Rewards')

    # Compute the rolling/moving average
    if len(rewards) >= window_size:
        # Cumulative sum trick for moving average
        cumsum = np.cumsum(np.insert(rewards, 0, 0))
        mov_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

        # Plot the moving average (shift the x-axis by window_size/2 for alignment)
        plt.plot(range(window_size, len(rewards) + 1),
                 mov_avg,
                 color='red',
                 linewidth=2,
                 label=f'Moving Average (window={window_size})')

    # Add axis labels and a title
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")

    # (Optional) Add grid lines
    plt.grid(True)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()




if __name__ == "__main__":
    env_name = "HalfCheetah-v4"
    agent = train_ppo(env_name)
    test_ppo(agent, env_name)
    plot_eps_rewards(agent)
