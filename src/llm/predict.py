"""
    DeepSeek Prompt 工程 — 零样本文本分类预测
    通过精心设计的 System Prompt 让 LLM 完成 10 分类任务
"""

import time
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
你是一个文本分类模型，负责完成**单标签多分类任务**。

# 任务说明

根据用户输入的短文本、标题或句子，判断其最可能所属的类别，并输出对应的类别名称。

# 可选类别

* finance
* realty
* stocks
* education
* science
* society
* politics
* sports
* game
* entertainment

# 分类要求

1. 输入为一段短文本、新闻标题或句子。
2. 从上述10个类别中选择最符合语义的一个类别。
3. 这是单标签分类任务，只能选择一个类别。
4. 不允许输出多个类别。
5. 不允许输出类别索引。
6. 不允许输出解释、分析过程、标点符号或其他任何内容。
7. 输出内容必须严格等于以上类别名称之一。

# 输出格式

仅输出类别名称，例如：
finance
或
sports

# 最终要求

无论输入内容是什么，最终输出必须且只能输出以下类别之一：
finance / realty / stocks / education / science / society / politics / sports / game / entertainment
"""

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


def call_deepseek_api(user_prompt: str, system_prompt: str = "You are a helpful assistant") -> str:
    """调用 DeepSeek API 获取回复"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    return response.choices[0].message.content


def predict_fun(data):
    """
    预测函数。

    Args:
        data: 字典，格式 {text: 文本内容}

    Returns:
        字典，格式 {text: 文本内容, pred_class: 预测类别名称}
    """
    text = data["text"]
    pred_class = call_deepseek_api(text, system_prompt=SYSTEM_PROMPT)
    data["pred_class"] = pred_class
    return data


if __name__ == "__main__":
    start_time = time.time()
    data = {"text": "今天股市大涨，沪指突破3000点"}
    result = predict_fun(data)
    print(result)
    print(f"推理延迟: {(time.time() - start_time) * 1000}ms")
