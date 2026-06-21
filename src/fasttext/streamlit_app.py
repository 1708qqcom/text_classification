"""
    FastText Streamlit 前端入口
    启动: streamlit run src/fasttext/streamlit_app.py
"""

from src.fasttext.config import FastTextConfig
from src.serving.streamlit_app import run_streamlit_page

config = FastTextConfig()
url = f"http://{config.api_host}:{config.api_port}/predict"

run_streamlit_page(
    buttons=[{"label": "FastText预测", "url": url}],
    title="投满分项目"
)
