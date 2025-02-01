import gym
import numpy as np
import torch as t
from state_indep_ppo import PPO  # or state_dep_ppo, either or.
import matplotlib.pyplot as plt

device = t.device("cuda" if t.cuda.is_available() else "cpu")
def train_ppo(env_name):
    env = gym.make(env_name)
    actionspace = env.action_space.shape[0]
    obsspace = env.observation_space.shape[0]
    print(f"actionspace: {actionspace}, obs space {obsspace}")
    agent = PPO(
        ob_space=obsspace,
        actions=actionspace,
        n_batches=10,
        gamma=0.95   ,
        lam=0.95,
        kl_coeff=0.2,
        clip_rewards=False,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0,
        a_lr=5e-4,
        c_lr=5e-4,
        device='cpu',
        max_ts=100,

        # Any custom kwargs can also be passed in here. For example:
        rollouts_per_batch=5,
        max_timesteps_per_episode=200,
        n_updates_per_iteration=3,
    )
    
    # 3. Train the agent
    total_timesteps = 2000//5  # Decide how long you want to train
    agent.learn(total_timesteps=total_timesteps, env=env)
    
    # 4. Close the environment
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


if __name__ == "__main__":
    env_name = "Pendulum"
    agent = train_ppo(env_name)
    test_ppo(agent, env_name)
    agent.logger.plot_eps_rewards()
