"""
    FastText FastAPI 服务入口
"""

from src.fasttext.predict import predict_fun
from src.fasttext.config import FastTextConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = FastTextConfig()
app = create_fastapi_app(
    predict_endpoints=[("/predict", "FastText预测", predict_fun)],
    title="FastText预测API",
    description="FastText模型预测接口"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
