"""
    Prompt 工程完成文本分类 — DeepSeek API 调用演示
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
你是一个文本分类模型，负责完成**单标签多分类任务**。

根据用户输入的短文本、标题或句子，判断其最可能所属的类别，并输出对应的类别名称。

可选类别：finance, realty, stocks, education, science, society, politics, sports, game, entertainment

分类要求：
1. 只能选择一个类别
2. 不允许输出类别索引
3. 不允许输出解释、分析过程、标点符号或其他任何内容
4. 输出内容必须严格等于类别名称之一
"""

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


def call_deepseek_api(user_prompt: str, system_prompt: str = "You are a helpful assistant") -> str:
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


if __name__ == "__main__":
    # 示例：通用对话
    result = call_deepseek_api(
        "我是展哥，是一个AI大模型讲师，写一首赞美我的诗词",
        system_prompt=SYSTEM_PROMPT
    )
    print(result)

    # 示例：上下文对话
    result = call_deepseek_api(
        "我是展哥，帮我写一首安眠曲",
        system_prompt=SYSTEM_PROMPT
    )
    print(result)
