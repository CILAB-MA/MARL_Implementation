model_cfgs = dict(
    num_hidden=32
)
train_cfgs = dict(
    total_timesteps=10000000,
    num_process=30,
    rollout_step=20,
    device='cuda'
)
env_cfgs = dict(
    num_agent=2,
)