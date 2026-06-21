"""
    BERT 微调方案 — 模型预测函数
"""

import time
import torch
from src.bert.config import BertConfig
from src.bert.train import MyBertModel
from transformers import BertTokenizer

config = BertConfig()

model = MyBertModel().to(config.device)
model.load_state_dict(
    torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
)

BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)


@torch.no_grad()
def predict_fun(data):
    """
    预测函数。

    Args:
        data: 字典，格式 {text: 文本内容}

    Returns:
        字典，格式 {text: 文本内容, pred_class: 预测类别名称}
    """
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
    data = {"text": "今天很安静，因为大家都没有说话，是不是生病了"}
    result = predict_fun(data)
    print(result)

    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    print(f"推理延迟: {(time.time() - start_time) / 100 * 1000}ms")
