"""
    BERT Prune FastAPI 服务入口
"""

from src.prune.predict import predict_fun
from src.prune.config import PruneConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = PruneConfig()
app = create_fastapi_app(
    predict_endpoints=[("/predict", "BERT剪枝预测", predict_fun)],
    title="BERT剪枝预测API",
    description="BERT剪枝模型预测接口"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
