"""
    BERT FastAPI 服务入口
"""

from src.bert.predict import predict_fun
from src.bert.config import BertConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = BertConfig()
app = create_fastapi_app(
    predict_endpoints=[("/predict", "BERT预测", predict_fun)],
    title="BERT预测API",
    description="BERT模型预测接口"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
