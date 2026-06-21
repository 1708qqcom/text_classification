"""
    BERT Distill FastAPI 服务入口（双路由：教师 + 学生）
"""

from src.distill.predict import predict_fun1, predict_fun2
from src.distill.config import DistillConfig
from src.serving.fastapi_app import create_fastapi_app, run_fastapi_app

config = DistillConfig()
app = create_fastapi_app(
    predict_endpoints=[
        ("/predict1", "教师模型", predict_fun1),
        ("/predict2", "学生模型", predict_fun2),
    ],
    title="BERT蒸馏预测API",
    description="BERT蒸馏模型预测接口（教师+学生对比）"
)

if __name__ == '__main__':
    run_fastapi_app(app, host=config.api_host, port=config.api_port)
