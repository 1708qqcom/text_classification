"""
    RF Streamlit 前端入口
    启动: streamlit run src/rf/streamlit_app.py
"""

from src.rf.config import RFConfig
from src.serving.streamlit_app import run_streamlit_page

config = RFConfig()
url = f"http://{config.api_host}:{config.api_port}/predict"

run_streamlit_page(
    buttons=[{"label": "随机森林预测", "url": url}],
    title="投满分项目"
)
