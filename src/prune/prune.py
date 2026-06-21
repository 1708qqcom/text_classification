"""
    BERT 模型剪枝 — 模块级非结构化剪枝
    对所有 Encoder 层的 attention.self.query 权重进行 L1 范数剪枝
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.nn.utils import prune
from src.prune.config import PruneConfig
from src.bert.train import MyBertModel, evaluate
from src.bert.dataset import build_dataloader
from transformers import BertTokenizer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = PruneConfig()
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)


def calculate_sparsity(model):
    """
    计算 query 线性层权重的稀疏度 = 0值参数数量 / 总参数量。
    """
    total_params = 0
    zero_params = 0
    num_encoder_layer = len(model.bert.encoder.layer)

    for i in range(num_encoder_layer):
        query_weight = model.bert.encoder.layer[i].attention.self.query.weight
        zero_params += (query_weight == 0).sum().item()
        total_params += query_weight.numel()

    return zero_params / total_params


def prune_model(model, amount=0.3):
    """
    模型剪枝函数，进行模块级非结构化剪枝。

    Args:
        model: 待剪枝模型
        amount: 剪枝比例

    Returns:
        剪枝后的模型
    """
    num_encoder_layers = len(model.bert.encoder.layer)
    prune_params = [
        (model.bert.encoder.layer[i].attention.self.query, "weight")
        for i in range(num_encoder_layers)
    ]

    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    for module, params in prune_params:
        prune.remove(module, params)

    torch.save(model.state_dict(), config.bert_prune_path)
    return model


if __name__ == "__main__":
    # 1. 加载模型
    model = MyBertModel().to(config.device)
    model.load_state_dict(
        torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
    )

    # 2. 计算剪枝前稀疏度
    sparsity = calculate_sparsity(model)
    print(f"剪枝前的稀疏度: {sparsity:.4f}")

    # 3. 构建数据加载器
    train_dataloader, val_dataloader, test_dataloader = build_dataloader(config, BERT_TOKENIZER)

    # 4. 剪枝
    model = prune_model(model, amount=0.3)

    # 5. 计算剪枝后稀疏度
    sparsity = calculate_sparsity(model)
    print(f"剪枝后的稀疏度: {sparsity:.4f}")

    # 6. 测试剪枝后模型
    test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
    print(f"剪枝后的模型测试结果: 准确率: {test_acc:.4f}, 测试集F1: {test_f1:.4f}")
