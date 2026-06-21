"""
    LLM FastAPI 服务入口
"""

from src.llm.predict import predict_fun
from src.common.base_config import BaseConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = BaseConfig()
app = create_fastapi_app(
    predict_endpoints=[("/predict", "LLM预测", predict_fun)],
    title="LLM预测API",
    description="DeepSeek Prompt工程预测接口"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
