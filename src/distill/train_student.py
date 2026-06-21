"""
    软标签蒸馏 — 学生模型训练
    Teacher: 微调后的 BERT-base-chinese
    Student: 小型 BERT 变体（hidden=256, layers=2）
    蒸馏损失 = (1-α)*CE + α*T²*KL
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig as HFBertConfig
from src.distill.config import DistillConfig
from src.bert.train import MyBertModel
from src.bert.dataset import build_dataloader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = DistillConfig()
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
BERT_CONFIG = HFBertConfig.from_pretrained(config.bert_path)
print(f"hidden_size: {BERT_CONFIG.hidden_size}")


class MyStudentModel(nn.Module):
    """
    软标签蒸馏的学生模型：小型 BERT + Linear
    hidden_size=256, num_hidden_layers=2, num_attention_heads=8
    """

    def __init__(self):
        super().__init__()
        self.student_config = HFBertConfig(
            vocab_size=BERT_CONFIG.vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=256 * 4,
            max_position_embeddings=config.max_len,
        )
        self.bert = BertModel(self.student_config)
        self.linear1 = nn.Linear(self.student_config.hidden_size, len(config.id2class))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear1(output.pooler_output)
        return logits


def train_one_epoch(teacher_model, student_model, train_dataloader,
                    ce_loss_fn, kl_loss_fn, optimizer):
    """训练一个轮次（蒸馏）"""
    student_model.train()
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []

    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids, attention_mask)

        student_logits = student_model(input_ids, attention_mask)

        ce_loss = ce_loss_fn(student_logits, labels)
        kl_loss = kl_loss_fn(
            F.log_softmax(student_logits / config.T, dim=-1),
            F.softmax(teacher_logits / config.T, dim=-1)
        )
        loss = (1 - config.alpha) * ce_loss + config.alpha * (config.T ** 2) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)
        y_preds = student_logits.argmax(dim=-1)
        total_preds.extend(y_preds.to("cpu").numpy().tolist())
        total_labels.extend(labels.to("cpu").numpy().tolist())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_student(teacher_model, student_model, val_dataloader,
                     ce_loss_fn, kl_loss_fn):
    """评估蒸馏模型"""
    student_model.eval()
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]

        teacher_logits = teacher_model(input_ids, attention_mask)
        student_logits = student_model(input_ids, attention_mask)

        ce_loss = ce_loss_fn(student_logits, labels)
        kl_loss = kl_loss_fn(
            F.log_softmax(student_logits / config.T, dim=-1),
            F.softmax(teacher_logits / config.T, dim=-1)
        )
        loss = (1 - config.alpha) * ce_loss + config.alpha * (config.T ** 2) * kl_loss

        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)
        y_preds = student_logits.argmax(dim=-1)
        total_preds.extend(y_preds.to("cpu").numpy().tolist())
        total_labels.extend(labels.to("cpu").numpy().tolist())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average="macro")
    return avg_loss, acc, f1


def train_student():
    """蒸馏训练主函数"""
    train_dataloader, val_dataloader, _ = build_dataloader(config, BERT_TOKENIZER)

    teacher_model = MyBertModel().to(config.device)
    teacher_model.load_state_dict(
        torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
    )
    teacher_model.eval()

    student_model = MyStudentModel().to(config.device)

    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s = [], [], []
    best_f1_score = 0.0

    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            teacher_model, student_model, train_dataloader,
            ce_loss_fn, kl_loss_fn, optimizer
        )
        val_loss, val_acc, val_f1 = evaluate_student(
            teacher_model, student_model, val_dataloader,
            ce_loss_fn, kl_loss_fn
        )

        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            torch.save(student_model.state_dict(), config.student_model_path)
            print(f"保存模型参数到：{config.student_model_path}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch: {epoch+1}/{config.epochs} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"train_f1: {train_f1:.4f} | val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | "
              f"time: {time.time()-start_time:.2f}s")

    return {
        "train_losses": train_losses, "train_accs": train_accs,
        "train_f1s": train_f1s,
        "val_losses": val_losses, "val_accs": val_accs, "val_f1s": val_f1s
    }


def plot_history(history):
    """可视化训练过程"""
    epochs = range(1, len(history['train_losses']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='train_loss')
    plt.plot(epochs, history['val_losses'], label='val_loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accs'], label='train_acc')
    plt.plot(epochs, history['val_accs'], label='val_acc')
    plt.title('Acc Curve'); plt.xlabel('Epoch'); plt.ylabel('Acc')
    plt.grid(True); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1s'], label='train_f1')
    plt.plot(epochs, history['val_f1s'], label='val_f1')
    plt.title('F1 Curve'); plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(config, BERT_TOKENIZER)

    # 训练（取消注释以执行）
    # results = train_student()
    # plot_history(results)

    # 测试
    teacher_model = MyBertModel().to(config.device)
    teacher_model.load_state_dict(
        torch.load(config.bert_model_path, weights_only=True, map_location=config.device)
    )
    student_model = MyStudentModel().to(config.device)
    student_model.load_state_dict(
        torch.load(config.student_model_path, weights_only=True, map_location=config.device)
    )

    test_loss, test_acc, test_f1 = evaluate_student(
        teacher_model, student_model, test_dataloader,
        ce_loss_fn=nn.CrossEntropyLoss(), kl_loss_fn=nn.KLDivLoss(reduction="batchmean")
    )
    print(f"模型测试结果：test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | test_f1: {test_f1:.4f}")
