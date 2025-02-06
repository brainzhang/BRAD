import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭 OneDNN 日志
os.environ["TF_TRT_ENABLE"] = "0"  # 禁用 TensorRT
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 TensorFlow GPU
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import gym
import yaml
import torch
import numpy as np
from collections import deque
import random
from models.dqn_model import DQN
from models.pretrained_model import ModelManager
from utils.logger import Logger
from Enviroments.cartpole_env import CartPoleEnvWrapper

class DQNAgent:
    def __init__(self, config_path):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # 初始化环境
        self.env = CartPoleEnvWrapper()
        
        # 创建DQN网络
        self.policy_net = DQN(
            self.config['model']['input_dim'],
            self.config['model']['hidden_dim'],
            self.config['model']['output_dim']
        )
        self.target_net = DQN(
            self.config['model']['input_dim'],
            self.config['model']['hidden_dim'],
            self.config['model']['output_dim']
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器和经验回放
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config['training']['lr']
        )
        self.memory = deque(maxlen=self.config['training']['buffer_size'])
        
        # 日志记录
        self.logger = Logger(self.config['logging']['log_dir'])
        
        self.epsilon = self.config['training']['epsilon_start']
    
    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return self.env.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state)).argmax().item()
    
def update_model(self):
    """执行DQN网络更新"""
    if len(self.memory) < self.config['training']['batch_size']:
        return

    # 从记忆库采样
    transitions = random.sample(self.memory, self.config['training']['batch_size'])
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
    
    # 转换为张量（修正部分）
    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)  # 修正行
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.BoolTensor(done_batch)
    
    # ...后续计算保持不变...