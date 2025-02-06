# **DQN 强化学习项目 README**

## **📌 项目简介**
本项目是一个基于 **深度 Q 网络（Deep Q-Network, DQN）** 的强化学习项目，主要用于 OpenAI Gym 中的 `CartPole-v1` 任务。DQN 通过**神经网络逼近 Q 值**，并使用**经验回放（Replay Buffer）**和**目标网络（Target Network）**提升训练稳定性。

---

## **📂 目录结构**
```bash
RL-DQN/
│── 📁 Config/                # 配置文件存放（超参数、环境设置等）
│   ├── hyperparameters.yaml  # DQN 训练参数
│
│── 📁 drl_env/               # Python 虚拟环境（如果你使用 `venv` 创建）
│   ├── bin/
│   ├── lib/
│   ├── ...
│
│── 📁 environments/          # 环境封装（Gym Wrappers、自定义环境）
│   ├── __init__.py
│   ├── cartpole_env.py       # 适用于 Gym 的 CartPole-v1 包装器
│
│── 📁 models/                # DQN 相关的神经网络模型
│   ├── __init__.py
│   ├── dqn_model.py          # DQN 主网络
│   ├── pretrained_model.py   # 预训练模型管理
│
│── 📁 utils/                 # 工具类（日志、数据处理等）
│   ├── __init__.py
│   ├── dataloader.py         # 数据加载（如果有离线数据）
│   ├── logger.py             # 训练日志记录
│
│── 📁 memory/                # 经验回放缓冲区（Replay Buffer）
│   ├── __init__.py
│   ├── replay_buffer.py      # 经验回放实现
│
│── 📁 agents/                # 强化学习 Agent 逻辑
│   ├── __init__.py
│   ├── dqn_agent.py          # DQN 智能体
│
│── 📁 experiments/           # 训练实验（不同的训练方案）
│   ├── __init__.py
│   ├── train_cartpole.py     # 训练 CartPole-v1
│   ├── train_custom_env.py   # 训练自定义环境
│
│── main.py                   # 项目主入口（可选）
│── train.py                   # 训练入口
│── test.py                    # 测试训练好的智能体
│── requirements.txt            # 依赖包列表
│── README.md                   # 项目文档
```

---

## **🔧 依赖安装**
### **1. 创建虚拟环境（可选）**
```bash
python -m venv drl_env
source drl_env/bin/activate  # Mac/Linux
# 或者在 Windows 上
# drl_env\Scripts\activate
```

### **2. 安装依赖**
```bash
pip install -r requirements.txt
```

如果你没有 `requirements.txt`，可以使用以下命令手动安装所需的 Python 库：
```bash
pip install gym numpy torch matplotlib pyyaml
```

---

## **🚀 训练 DQN 代理**
### **1. 运行训练脚本**
```bash
python train.py
```

### **2. 训练参数（`Config/hyperparameters.yaml`）**
```yaml
training:
  total_episodes: 1000
  batch_size: 64
  gamma: 0.99
  lr: 0.001
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  target_update: 10
  buffer_size: 10000

model:
  input_dim: 4
  hidden_dim: 128
  output_dim: 2

logging:
  log_dir: ./logs
  log_interval: 20
```

### **3. 训练日志**
训练过程中，你会在 `./logs/` 目录下找到日志文件，方便可视化训练进度。

---

## **🛠 代码说明**
### **1. 训练脚本 `train.py`**
```python
import yaml
from agents.dqn_agent import DQNAgent
from environments.cartpole_env import CartPoleEnvWrapper

# 读取超参数
with open("Config/hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

env = CartPoleEnvWrapper()
agent = DQNAgent(config)

for episode in range(config['training']['total_episodes']):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

    print(f"Episode {episode}: Total Reward = {total_reward}")
```

### **2. 经验回放 `memory/replay_buffer.py`**
```python
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)
```

---

## **🧪 测试智能体**
### **1. 运行测试脚本**
```bash
python test.py
```

### **2. 评估训练好的智能体**
```python
import torch
from agents.dqn_agent import DQNAgent
from environments.cartpole_env import CartPoleEnvWrapper

# 加载训练好的模型
env = CartPoleEnvWrapper()
agent = DQNAgent("Config/hyperparameters.yaml")
agent.policy_net.load_state_dict(torch.load("./models/dqn_trained.pth"))

state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    state, _, done, _ = env.step(action)
    env.env.render()
```

---

## **📜 贡献 & 许可**
本项目开源，遵循 **MIT License**，欢迎贡献代码！

```bash
git clone https://github.com/your-repo/RL-DQN.git
cd RL-DQN
git checkout -b new-feature-branch
git commit -m "Add new feature"
git push origin new-feature-branch
```

**欢迎提交 Pull Request 或 Issue 🚀！**

