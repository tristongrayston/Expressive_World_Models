CUCAI Project 2025:

Contributors: 


# Relevant Papers: 
For those interested in helping out with the project, these papers provide the theory which drives our experiments. 
That being said, there are more ways to contribute that involve runtime optimization, GPU computing, etc.
Shoot me a message if you're interested. 
\
[PPO](https://arxiv.org/abs/1707.06347) \
[General Advantage Esimtation](https://arxiv.org/pdf/1506.02438) \
[Neural ODES](https://arxiv.org/pdf/1506.02438) \
[LTC Networks](https://arxiv.org/abs/2006.04439) \
[NCPS](https://www.nature.com/articles/s42256-020-00237-3) \
[CFCS](https://www.nature.com/articles/s42256-022-00556-7) 

# Environments 

## Pendulum
We have two implementations of PPO: PPO with state-dependent variance, and with state-independent variance. 

### State Indep Hyperparameters:
gamma: 0.95
lam: 0.95
kl_coeff: 0.2
clip_rewards: False
clip_param: 0.2
vf_clip_param: 10.0
entropy_coeff: 0
a_lr: 5e-4
c_lr: 5e-4
device: cpu
max_ts: 100
rollouts_per_batch: 5
max_timesteps_per_episode: 200
n_updates_per_iteration: 3

### State Dep Hyperparameters
n_batches: 10
gamma: 0.95
lam: 0.95
kl_coeff: 0.2
clip_rewards: False
clip_param: 0.2
vf_clip_param: 10.0
entropy_coeff: 0
a_lr: 1e-3
c_lr: 1e-3
device: 'cpu'
max_ts: 100
rollouts_per_batch: 4
max_timesteps_per_episode: 200
n_updates_per_iteration: 2