"""
    FastText 方案 — 模型预测函数（默认使用 char_1 模型）
"""

import fasttext
import time
import jieba
from src.fasttext.config import FastTextConfig
import warnings
warnings.filterwarnings("ignore")

import numpy as np
old_array = np.array

def patched_array(obj, copy=True, *args, **kwargs):
    if copy is False:
        copy = True
    return old_array(obj, copy=copy, *args, **kwargs)

np.array = patched_array

config = FastTextConfig()
model = fasttext.load_model(config.fasttext_model_path.replace(".bin", "_char_1.bin"))


def predict_fun(data):
    """
    预测函数。

    Args:
        data: 字典，格式 {text: 文本内容}

    Returns:
        字典，格式 {text: 文本内容, pred_class: 预测类别名称}
    """
    text = data["text"]
    text_split = " ".join(list(text))
    y_pred = model.predict(text_split)[0][0]
    pred_class = y_pred.replace("__label__", "")
    data["pred_class"] = pred_class
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
