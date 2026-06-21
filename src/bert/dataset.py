"""
    BERT 数据管道 — Dataset、collate_fn、DataLoader 构建
    供 bert/distill/prune 复用
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def load_raw_data(file_path):
    """
    加载数据文件，返回列表 [(文本, 标签索引)]。

    Args:
        file_path: 原始文件路径

    Returns:
        list of (text, label_index)
    """
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            results.append((text, int(label)))
    return results


class MyDataset(Dataset):
    """自定义数据集类，构造单个样本的输入和输出"""

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        text, label = self.data_list[index]
        return text, label


def create_collate_fn(tokenizer, max_len):
    """
    创建 collate_fn 批次整理函数。

    Args:
        tokenizer: BERT 分词器
        max_len: 最大序列长度

    Returns:
        collate_fn 函数
    """
    def collate_fn(batch):
        texts, labels = zip(*batch)
        output = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len
        )
        input_ids = torch.tensor(output.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(output.attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, labels
    return collate_fn


def build_dataloader(config, tokenizer):
    """
    构建训练/验证/测试数据加载器。

    Args:
        config: BERT 配置对象
        tokenizer: BERT 分词器

    Returns:
        (train_dataloader, dev_dataloader, test_dataloader)
    """
    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)

    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)

    collate = create_collate_fn(tokenizer, config.max_len)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate
    )
    return train_dataloader, dev_dataloader, test_dataloader
