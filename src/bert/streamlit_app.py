"""
    BERT Streamlit 前端入口
    启动: streamlit run src/bert/streamlit_app.py
"""

from src.bert.config import BertConfig
from src.serving.streamlit_app import run_streamlit_page

config = BertConfig()
url = f"http://{config.api_host}:{config.api_port}/predict"

run_streamlit_page(
    buttons=[{"label": "BERT预测", "url": url}],
    title="投满分项目"
)
