"""
    通用 API 测试客户端
    支持交互式输入文本并调用任意预测接口
"""

import requests


def test_api(host="127.0.0.1", port=5000, route="/predict"):
    """
    交互式测试 API 服务。

    Args:
        host: API 服务地址
        port: API 服务端口
        route: API 路由路径
    """
    url = f"http://{host}:{port}{route}"
    print(f"测试地址: {url}")

    try:
        text = input("请输入文本内容：")
        data = {"text": text}
        r = requests.post(url, json=data)
        print(f"预测结果：{r.json()}")
    except Exception as e:
        print(f"出问题了，请联系管理员, {e}!")


if __name__ == "__main__":
    test_api()
