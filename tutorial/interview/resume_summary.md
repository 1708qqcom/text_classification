# 投满分项目 — 工作内容描述

## 核心职责与成果

### 1. 基线搭建

主导传统机器学习与深度学习多方案基线搭建，完成 4 类模型共 6 条技术路线的训练与横向对比：
- **TF-IDF + 随机森林**：jieba 分词 + 停用词过滤 + TF-IDF 向量化，n_estimators=100，作为传统机器学习 baseline
- **FastText**：字符级/词级两种分词粒度 × 默认参数/自动调参（autotune），共 4 组对照实验，覆盖轻量级深度模型方案
- **BERT-base-chinese 微调**：基于 HuggingFace Transformers，独立搭建 `MyDataset → collate_fn → DataLoader` 数据流，使用 `pooler_output` + 线性分类头完成下游分类适配
- **DeepSeek LLM Prompt 工程**：设计约束型系统提示词，基于 OpenAI SDK 调用 DeepSeek API 实现零样本分类，作为少样本/冷启动场景的技术储备

产出多模型精度-速度-资源对比矩阵，明确各方案适用边界。

### 2. 模型调优

针对 BERT 微调进行多维度超参优化，解决训练不稳定与显存受限问题：
- 学习率（1e-5）、权重衰减（1e-2）、批次大小（64）、最大序列长度（32）、训练轮次（5 epochs）等关键参数调优
- 使用 AdamW 优化器 + CrossEntropyLoss，基于验证集 F1 保存最优 checkpoint
- 调优后测试集 F1 达到目标水平，提升约 3 个百分点

### 3. 模型压缩优化

设计并实现 BERT 模型的三条轻量化路线，满足工业级端侧部署要求：
- **动态量化（DQ）**：`torch.quantization.quantize_dynamic` 对 Linear 层 FP32 → INT8 量化，体积压缩约 4×，推理延迟降低约 2×
- **知识蒸馏（KD）**：搭建 Teacher-Student 框架，以微调 BERT 为 Teacher，通过软标签蒸馏将知识迁移至小模型，体积压缩约 3×，精度损失可控
- **权重剪枝（Pruning）**：按比例裁剪低权重连接实现模型稀疏化，体积压缩约 2×

### 4. 工程化落地辅助

配合完成模型部署与前端上线，支撑 API 服务与 Demo 快速交付：
- 为每条技术路线编写统一接口的 `predict_fun(data)` 推理函数，输入 `{"text": str}` / 输出 `{"text": str, "pred_class": str}` 格式标准化
- 基于 FastAPI + Pydantic 构建 RESTful API（`POST /predict`），支持请求体校验与响应模型约束；同步提供 Flask 版本覆盖不同部署环境
- 基于 Streamlit 搭建 Web 交互前端，支持文本输入 → 实时预测 → 推理延迟展示
