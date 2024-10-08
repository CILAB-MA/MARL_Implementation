model_cfgs = dict(
    algo_name = "QMIX",
    hidden_dim=[64,64],
    embed_dim = 64,
    # embed_dim = 1,
    hypernet_layers = 2,
    # hypernet_layers = 1,
    hypernet_embed = 32,
    lr = 3.e-4,
)
train_cfgs = dict(
    total_timesteps=10_000_000,
    replay_buffer_size=2_000_000,
    # replay_buffer_size=1_000,
    num_process = 16,
    # num_process = 4,
    epsilon=1.0,
    epsilon_decay_steps=300_000,
    epsilon_min=0.05,
    epsilon_decay_interval=1000,
    gamma=0.99,
    target_update_interval=200,
    batch_size=128,
    # batch_size=8,
    test_interval=10000,
    save_interval=50000,
    use_wandb=True,
)
env_cfgs = dict(
    num_agent=2,
    central_reward=False,
)