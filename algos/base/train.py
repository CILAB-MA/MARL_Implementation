from tqdm import tqdm

def train(cfgs):
    train_cfgs = cfgs.train_cfgs
    print(train_cfgs)
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    t = 0
    for _ in tqdm(range(train_cfgs['total_timesteps'])):
        t += 1
    print('Train completed!!')

