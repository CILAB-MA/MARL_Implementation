from algos.base.agent import BaseAgent
from algos.base.train import train as basetrainer
from algos.base import cfg as basecfg
AGENT = dict(base=BaseAgent)
TRAINER = dict(base=basetrainer)
CFGS = dict(base=basecfg)