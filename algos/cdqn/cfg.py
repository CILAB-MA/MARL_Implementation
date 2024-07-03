model_cfgs = dict(
    obs_dim=5,
    hidden_dim=128,
    action_dim=5,
    epsilon=1.0,
    epsilon_decay=0.99,
    lr=1e-3,
    gamma=0.99,
    batch_size=32,
    buffer_size=512,
)
train_cfgs = dict(
    device='cpu',
    n_episodes=40000,
    target_update_freq=50
)
env_cfgs = dict(
    action_space=5,
    observation_space=[],
    num_agent=1,
)