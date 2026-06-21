"""
    BERT 蒸馏方案 — 模型预测函数（教师 + 学生双模型）
"""

import time
import torch
from src.distill.config import DistillConfig
from src.bert.train import MyBertModel
from src.distill.train_student import MyStudentModel
from transformers import BertTokenizer

config = DistillConfig()
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)

# 加载教师模型
teacher_model = MyBertModel().to(config.device)
teacher_model.load_state_dict(
    torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
)

# 加载学生模型
student_model = MyStudentModel().to(config.device)
student_model.load_state_dict(
    torch.load(config.student_model_path, weights_only=True, map_location=config.device)
)


def _predict(model, text):
    """通用预测逻辑"""
    model.eval()
    output = BERT_TOKENIZER(
        [text], padding=True, truncation=True, max_length=config.max_len
    )
    logits = model(
        input_ids=torch.tensor(output.input_ids).to(config.device),
        attention_mask=torch.tensor(output.attention_mask).to(config.device),
    )
    y_pred = logits.argmax(dim=-1)
    return config.id2class[y_pred[0].item()]


@torch.no_grad()
def predict_fun1(data):
    """教师模型预测"""
    data["pred_class"] = _predict(teacher_model, data["text"])
    return data


@torch.no_grad()
def predict_fun2(data):
    """学生模型预测"""
    data["pred_class"] = _predict(student_model, data["text"])
    return data


if __name__ == "__main__":
    data = {"text": "今天股市大涨，沪指突破3000点"}

    # 教师模型
    result = predict_fun1(data)
    print(f"教师模型: {result}")
    start_time = time.time()
    for i in range(100):
        predict_fun1(data)
    print(f"教师模型推理延迟: {(time.time() - start_time) / 100 * 1000}ms")

    # 学生模型
    result = predict_fun2(data)
    print(f"学生模型: {result}")
    start_time = time.time()
    for i in range(100):
        predict_fun2(data)
    print(f"学生模型推理延迟: {(time.time() - start_time) / 100 * 1000}ms")
