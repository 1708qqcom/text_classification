"""
    通用 FastAPI 工厂函数
    支持单路由和多路由模式
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    pred_class: str


def create_fastapi_app(predict_endpoints, title="预测API", description="模型预测接口"):
    """
    创建 FastAPI 应用。

    Args:
        predict_endpoints: list of (route, label, predict_fn)
            例: [("/predict", "RF预测", rf_predict)]
            例: [("/predict1", "教师模型", teacher_predict),
                 ("/predict2", "学生模型", student_predict)]
        title: API 标题
        description: API 描述

    Returns:
        FastAPI app 实例（未启动）
    """
    app = FastAPI(title=title, description=description)

    for route, label, predict_fn in predict_endpoints:
        def make_endpoint(fn=predict_fn, lbl=label):
            def endpoint(request: PredictRequest):
                data = {"text": request.text}
                print(f"[{lbl}] data: {data}")
                result = fn(data)
                return result
            return endpoint

        # 为每个路由生成唯一的函数名
        func_name = route.replace('/', '_').lstrip('_')
        make_endpoint.__name__ = func_name
        app.post(route, response_model=PredictResponse)(make_endpoint())

    return app


def run_fastapi_app(app, host="127.0.0.1", port=5000):
    """启动 FastAPI 应用"""
    uvicorn.run(app, host=host, port=port)
