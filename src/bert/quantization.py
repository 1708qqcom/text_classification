"""
    BERT 模型动态量化（DQ）— FP32 → INT8
    仅支持 CPU
"""

import torch
from torch import nn
from src.bert.config import BertConfig
from src.bert.train import MyBertModel, evaluate
from src.bert.dataset import build_dataloader
from transformers import BertTokenizer

print(f"当前量化引擎：{torch.backends.quantized.engine}")
print(f"支持的量化引擎：{torch.backends.quantized.supported_engines}")

config = BertConfig()
print(f"当前设备：{config.device}")

BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)

# 构建数据加载器
train_dataloader, dev_dataloader, test_dataloader = build_dataloader(config, BERT_TOKENIZER)

# 加载原始模型（FP32）
model = MyBertModel().to(config.device)
model.load_state_dict(
    torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
)

# 评估原始模型
test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
print(f"原始模型评估结果：loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")

# 动态量化
model.eval()
model_quant = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 评估量化后模型
test_loss, test_acc, test_f1 = evaluate(
    model_quant, test_dataloader, loss_fn=nn.CrossEntropyLoss()
)
print(f"量化后模型评估结果：loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")

# 保存量化模型
torch.save(model_quant.state_dict(), config.bert_quantization_path)
print(f"保存量化模型成功：{config.bert_quantization_path}")
