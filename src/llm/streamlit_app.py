"""
    LLM Streamlit 前端入口
    启动: streamlit run src/llm/streamlit_app.py
"""

from src.common.base_config import BaseConfig
from src.serving.streamlit_app import run_streamlit_page

config = BaseConfig()
url = f"http://{config.api_host}:{config.api_port}/predict"

run_streamlit_page(
    buttons=[{"label": "DeepSeek预测", "url": url}],
    title="投满分项目"
)
