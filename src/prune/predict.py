"""
    BERT 剪枝方案 — 模型预测函数
    与 src/bert/predict.py 逻辑一致，使用剪枝后的模型权重。
    通过 import src.bert.predict 复用预测逻辑，仅更换模型路径。
"""

import time
import torch
from src.prune.config import PruneConfig
from src.bert.train import MyBertModel
from transformers import BertTokenizer

config = PruneConfig()

model = MyBertModel().to(config.device)
model.load_state_dict(
    torch.load(config.bert_prune_path, weights_only=True, map_location=config.device)
)

BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)


@torch.no_grad()
def predict_fun(data):
    """预测函数"""
    model.eval()
    text = data["text"]
    output = BERT_TOKENIZER(
        [text], padding=True, truncation=True, max_length=config.max_len
    )
    logits = model(
        input_ids=torch.tensor(output.input_ids).to(config.device),
        attention_mask=torch.tensor(output.attention_mask).to(config.device),
    )
    y_pred = logits.argmax(dim=-1)
    data["pred_class"] = config.id2class[y_pred[0].item()]
    return data


if __name__ == "__main__":
    data = {"text": "今天股市大涨，沪指突破3000点"}
    result = predict_fun(data)
    print(result)

    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    print(f"推理延迟: {(time.time() - start_time) / 100 * 1000}ms")
