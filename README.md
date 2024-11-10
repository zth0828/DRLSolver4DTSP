# Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning
This code solves dynamic trvaeling saleman problem with deep reinforcement learning. For more details, please see our paper [Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9537638) which has been accepted at IEEE-TNNLS. If this code is useful for your work, please cite our paper:

```
@article{zhang2021solving,
  title={Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning},
  author={Zhang, Zizhen and Liu, Hong and Zhou, MengChu and Wang, Jiahai},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```

## Dependencies

* python = 3.6.3
* NumPy
* Scipy
* PyTorch = 1.7
* tensorboard_logger

## Quick Start
# 对于使用 19 个客户训练 DTSP 实例，并使用部署作为模型 M1 的 REINFORCE 基线：
For training DTSP instances with 19 customers and using rollout as REINFORCE baseline with model M1:

```
python m1/train.py --baseline rollout --graph_size 19
```
# 对于使用 19 个客户训练 DTSP 实例，并使用推出作为模型 M2 的 REINFORCE 基线：
For training DTSP instances with 19 customers and using rollout as REINFORCE baseline with model M2:

```
python m2/train.py --baseline rollout  --graph_size 19

# python m2/train.py --baseline rollout  --graph_size 19 --resume .\outputs\tsp_19\run_20240722T182436\epoch-8.pt
```
# 对于使用经过训练的模型 M1 测试具有 19 个客户的 DTSP 实例：
For testing DTSP instances with 19 customers with a trained model M1:

```
python m1/test.py --baseline rollout --graph_size 19 --resume trained_models/m1/normal_19.pt
```

# 对于使用经过训练的模型 M2 测试具有 19 个客户的 DTSP 实例：
For testing DTSP instances with 19 customers with a trained model M2:

```
python m2/test.py --baseline rollout --graph_size 19 --resume trained_models/m2/normal_19.pt
```

## Acknowledgements

Our code is adpated from [https://github.com/wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route).
