"""
    通用 Streamlit 前端组件
    支持单按钮和多按钮布局（如蒸馏路线需要双按钮对比）
"""

import streamlit as st
import requests
import time


def run_streamlit_page(buttons, title="投满分项目"):
    """
    运行 Streamlit 预测页面。

    Args:
        buttons: list of dict with keys:
            - label: 按钮文字，如 "随机森林预测"
            - url:  API 端点 URL，如 "http://127.0.0.1:5000/predict"
        title: 页面标题
    """
    st.title(title)
    text = st.text_input("请输入文本：")

    n_buttons = len(buttons)

    if n_buttons == 1:
        _render_single_button(buttons[0], text)
    else:
        cols = st.columns(n_buttons)
        for i, btn in enumerate(buttons):
            with cols[i]:
                _render_single_button(btn, text)


def _render_single_button(btn, text):
    """渲染单个预测按钮"""
    if st.button(btn["label"]):
        try:
            start_time = time.time()
            data = {"text": text}
            r = requests.post(btn["url"], json=data)
            total_time = (time.time() - start_time) * 1000
            print(f"预测结果：{r.json()}")
            st.success(
                f"预测结果：{r.json()['pred_class']} | "
                f"推理时长：{total_time:.2f}ms"
            )
        except Exception as e:
            st.error("出问题了，请联系管理员!")
            print(f"出问题了，请联系管理员, {e}!")
