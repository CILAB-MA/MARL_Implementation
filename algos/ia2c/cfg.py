model_cfgs = dict(
    num_hidden=32,
    num_agent=2
)
train_cfgs = dict(
    total_timesteps=10000000,
    num_process=8,
    rollout_step=20,
    device='cuda'
)
env_cfgs = dict(
    num_agent=2,
)