# MARL_Implementation
Implementation study of MARL book(https://www.marl-book.com/)
# Environment
Robot Warehouse (https://github.com/semitable/robotic-warehouse)
# Installation
```
docker pull kevinjeon119/marl-book:v1
```
# Get Started
For testing environment,
```
python test/test_env.py
```
For ur algo, register your algo in `algos/__init__.py` and args in `main.py`
```
python main.py --trainer-name YOUR_ALGOS
```
# Implementation List
- [ ] CDQN
- [ ] IDQN
- [ ] IA2C
- [ ] CA2C
- [ ] COMA
- [ ] VDN
- [ ] QMIX
- [ ] QTRAN-base
- [ ] Deep Joint Action Learning
- [ ] Encoder-decoder Action Learning
- [ ] DQN + Experience Sharing
- [ ] PSRO
