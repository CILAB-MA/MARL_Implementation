from algos import TRAINER
from algos import CFGS
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description='This is for common configs. Do not put unique configs (e.g. ppo-epoch)')
    parser.add_argument('--trainer-name', default='base', type=str, choices=['base', 'idqn', 'cdqn', 'ia2c', 'ca2c'])
    return parser.parse_args()


def main(args):
    cfgs = CFGS[args.trainer_name]
    TRAINER[args.trainer_name](cfgs)


if __name__ == '__main__':
    args = parse_args()
    main(args)
