
model_cfgs = dict(
    optimizer="Adam",
    lr=0.0003,
    grad_clip=False,
    gamma=0.99,
    entropy_coef=0.001,
    value_loss_coef=0.5,
    centralised=False
)

train_cfgs = dict(
    use_wandb=False,
    total_timesteps=10,
    num_process=8
)

env_cfgs = dict(
    n_steps=10,
    total_steps=20_050_000,
    eval_interval_steps=10_000,
    eval_episodes=10_000,
)