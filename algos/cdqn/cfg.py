model_cfgs = dict(
    obs_dim=5,
    hidden_dim=128,
    action_dim=5,
    epsilon=1.0,
    epsilon_decay=0.99,
    lr=3e-4,
    gamma=0.99,
    batch_size=128,
    buffer_size=100000,
    target_update_freq=200,
)
train_cfgs = dict(
    device='cpu',
    n_episodes=20000,
    num_process=8,
    use_wandb=True,
)
env_cfgs = dict(
    action_space=[],
    observation_space=[],
    num_agent=1,
)

