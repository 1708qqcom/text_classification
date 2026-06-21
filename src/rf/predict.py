"""
    随机森林方案 — 模型预测函数
"""

import jieba
from src.rf.config import RFConfig
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

config = RFConfig()

# 加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


def predict_fun(data):
    """
    模型预测函数。

    Args:
        data: 字典，格式 {text: 文本内容}

    Returns:
        字典，格式 {text: 文本内容, pred_class: 预测类别名称}
    """
    text = data["text"]
    words = " ".join(jieba.lcut(text)[:40])
    features = tfidf_vectorizer.transform([words])
    y_pred = rf_model.predict(features)[0]
    y_pred_class = config.id2class[y_pred]
    data["pred_class"] = y_pred_class
    return data


if __name__ == "__main__":
    data = {
        "text": "今天股市大涨，沪指突破3000点"
    }
    result = predict_fun(data)
    print(result)

    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    print(f"推理延迟: {(time.time() - start_time) / 100 * 1000}ms")
