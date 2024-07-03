model_cfgs = dict(
    hidden_dim=128,
    lr = 0.001,
)
train_cfgs = dict(
    total_timesteps=10,
    replay_buffer_size=100,
    num_process = 5,
    epsilon=0.1,
    gamma=0.99,
    
)
env_cfgs = dict(
    num_agent=2,
)