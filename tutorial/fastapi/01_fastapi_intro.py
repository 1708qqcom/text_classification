"""
演示：
    FastAPI 框架入门操作，创建一个简单的 API 服务

说明：
    1) 安装依赖：pip install fastapi uvicorn
    2) app = FastAPI()   # 创建 FastAPI 应用
    3) @app.get("/")     # 创建 GET 路由
    4) uvicorn.run(...)  # 启动服务
FastAPI与Flask的区别:
    1) 使用方法：
        fastapi:
            from fastapi import FastAPI
            app = FastAPI()
            @app.get("/")
            def predict():
                return "hello world"
        flask:
            from flask import Flask
            app = Flask(__name__)
            @app.route('/', methods=['GET'])
            def predict():
                return "hello world"
    2) 特点:
        fastapi: ASGI 异步服务器, 性能更高
        flask: WSGI 同步服务器, 异步较弱
"""

# FastAPI: 用于创建 Web API
from fastapi import FastAPI
# FileResponse: 用于返回文件（例如图片）
from fastapi.responses import FileResponse
# uvicorn: FastAPI 常用的运行服务器
import uvicorn

# 1. 创建 FastAPI 应用
app = FastAPI(
    title="FastAPI入门演示",
    description="FastAPI入门的课堂示例",
    version="1.0.0"
)

# 2. 创建路由（接口）
# 与 Flask 的 @app.route('/', methods=['GET']) 类似，这里直接写成 @app.get("/")
@app.get("/")
def predict():
    """
    视图函数（接口处理函数）：
        访问根路径 / 时，返回一张图片文件。
    """
    # 返回本地图片文件，效果与 Flask 示例中的 send_file 类似
    # 请确保当前脚本同级目录下有 imgs/img.jpg
    # return "hello world"
    return FileResponse(path="img.jpg", media_type="image/jpeg")


# 3. 启动服务
if __name__ == "__main__":
    # host='127.0.0.1'：只允许本机访问
    # port=8000：服务端口（FastAPI 常用 8000）
    # reload=True：代码修改后自动重启
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # 如果你想让局域网设备访问，可改成：
    # uvicorn.run(app, host="0.0.0.0", port=8000)
