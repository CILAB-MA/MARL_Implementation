
model_cfgs = dict(
    optimizer="Adam",
    lr=3.e-4,
    grad_clip=False,
    gamma=0.99,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    centralised=True
)

train_cfgs = dict(
    total_timesteps=10,
    num_process=8
)

env_cfgs = dict(
    n_steps=10,
    total_steps=100_000,
    eval_interval_steps=10_000,
    eval_episodes=10_000,
)