"""
    RF FastAPI 服务入口
    启动: python -m src.rf.fastapi_server
"""

from src.rf.predict import predict_fun
from src.rf.config import RFConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = RFConfig()
app = create_fastapi_app(
    predict_endpoints=[("/predict", "RF预测", predict_fun)],
    title="RF预测API",
    description="随机森林模型预测接口"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
