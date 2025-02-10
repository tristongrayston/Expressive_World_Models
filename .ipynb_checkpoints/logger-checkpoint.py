import numpy as np
import time
import matplotlib.pyplot as plt


class logger():
    def __init__(self, window_size):
        self.delta_t = time.time_ns()
        self.t_so_far = 0           # timesteps so far
        self.i_so_far = 0           # iterations so far
        self.batch_lens = []        # episodic lengths in batch
        self.batch_rews = []        # episodic returns in batch
        self.actor_losses = []      # losses of actor network in current iteration
        self.raw_eps_rewards = []   # A track of the sum of rewards for all episodes.
        self.eps_rolling = [0.0]
        self.rolling_avg = 0.0
        self.window_size = window_size

    def change_rolling_average(self):
        cumsum = np.cumsum(np.insert(self.raw_eps_rewards, 0, 0))
        mov_avg = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / float(self.window_size)
        if len(mov_avg) == 0:
            self.rolling_avg = 0.0
        else:
            self.rolling_avg = mov_avg[-1]

    def plot_eps_rewards(self, window_size=10):
        """
        Plots episode rewards over time and includes a moving average trend line.
        
        Parameters:
        -----------
        agent : object
            Your agent object, which should have a logger dict containing 'eps_rewards'.
        window_size : int
            The size of the window over which to compute the moving average.
        """

        # Create a new figure
        plt.figure(figsize=(8, 6))

        # Plot the raw episode rewards
        plt.plot(self.raw_eps_rewards, marker='o', linestyle='-', color='b', label='Episode Rewards')
   
        # Compute the rolling/moving average

        if len(self.raw_eps_rewards) >= window_size:
            # Cumulative sum trick for moving average
            cumsum = np.cumsum(np.insert(self.raw_eps_rewards, 0, 0))
            mov_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

            # Plot the moving average (shift the x-axis by window_size/2 for alignment)
            plt.plot(range(window_size, len(self.raw_eps_rewards) + 1),
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
