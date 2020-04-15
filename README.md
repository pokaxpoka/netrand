# Introduction
This repository implements [Lee et al. Network Randomization: A Simple Technique for Generalization in Deep Reinforcement Learning. In ICLR, 2020](https://arxiv.org/abs/1910.05396) in TensorFlow==1.12.0.
```
@inproceedings{lee2020network,
  title={Network Randomization: A Simple Technique for Generalization in Deep Reinforcement Learning},
  author={Lee, Kimin and Lee, Kibok and and Shin, Jinwoo and Lee, Honglak},
  booktitle={ICLR},
  year={2020}
}
```

# Preliminaries
This code is based on [CoinRun](https://github.com/openai/coinrun) platform. 
After installing all packages in [CoinRun](https://github.com/openai/coinrun), 
replace `coinrun.cpp`, `coinrunenv.py`, `config.py`, `policies.py`, `ppo2.py`, `random_ppo2.py`, and `train_random.py` with files in `sources/` in this repository.

# Training PPO agents

Vanilar PPO 
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500
```

Vanilar PPO + CutOut
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -uda 1
```

Vanilar PPO + Dropout
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -dropout 0.1
```

Vanilar PPO + BN
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -norm 1
```

Vanilar PPO + L2
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -l2 0.0001
```

Vanilar PPO + Grayout
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -ubw 1
```


Vanilar PPO + Inversion
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -ui 1
```


Vanilar PPO + Color Jitter
```
python -m coinrun.train_agent --run-id myrun --save-interval 1 --num_levels 500 -uct 1
```

PPO + ours
```
python -m coinrun.train_random --run-id myrun --save-interval 1 --num_levels 500 -lstm 2
```

# Test on unseen environments

```
python -m coinrun.enjoy --test-eval --restore-id myrun -num-eval N -rep K -train_flag 1
```
