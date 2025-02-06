import torch
import os

class ModelManager:
    @staticmethod
    def save_model(model, path="models/pretrained", filename="dqn_cartpole.pth"):
        """保存训练好的模型"""
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, filename))
        print(f"Model saved to {os.path.join(path, filename)}")

    @staticmethod
    def load_model(model, path="models/pretrained/dqn_cartpole.pth"):
        """加载预训练模型"""
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            model.eval()
            print("Pretrained model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"No pretrained model found at {path}")