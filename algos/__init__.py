from algos.base.agent import BaseAgent
from algos.base.train import train as basetrainer
from algos.base import cfg as basecfg

from algos.idqn.agent import IDQNAgent
from algos.idqn.train import train as idqntrainer
from algos.idqn import cfg as idqncfg

from algos.cdqn.agent import CDQNAgent
from algos.cdqn.train import train as cdqntrainer
from algos.cdqn import cfg as cdqncfg

from algos.ia2c.agent import IA2CAgent
from algos.ia2c.train import train as ia2ctrainer
from algos.ia2c import cfg as ia2ccfg

from algos.ca2c.agent import CA2CAgent
from algos.ca2c.train import train as ca2ctrainer
from algos.ca2c import cfg as ca2ccfg

from algos.qtran.agent import QTRANAgent
from algos.qtran.train import train as qtrantrainer
from algos.qtran import cfg as qtrancfg


AGENT = dict(base=BaseAgent, idqn=IDQNAgent, cdqn=CDQNAgent, ia2c=IA2CAgent, ca2c=CA2CAgent,
             qtran=QTRANAgent)
TRAINER = dict(base=basetrainer, idqn=idqntrainer, cdqn=cdqntrainer, ia2c=ia2ctrainer, ca2c=ca2ctrainer,
               qtran=qtrantrainer)
CFGS = dict(base=basecfg, idqn=idqncfg, cdqn=cdqncfg, ia2c=ia2ccfg, ca2c=ca2ccfg, qtran=qtrancfg)