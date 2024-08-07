model_cfgs = dict(
    algo_name = "IDQN",
    hidden_dim=[64,64],
    lr = 3.e-4,
)
train_cfgs = dict(
    total_timesteps=10000000,
    replay_buffer_size=2000000,
    num_process = 16,
    epsilon=1.0,
    epsilon_decay_steps=300000,
    epsilon_min=0.05,
    epsilon_decay_interval=1000,
    gamma=0.99,
    target_update_interval=200,
    batch_size=128,
    test_interval=10000,
    save_interval=50000,
    use_wandb=True,
)
env_cfgs = dict(
    num_agent=2,
    central_reward=False,
)