import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import argparse

# 数据集类
class ChineseTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        
        # 获取 input_ids 和 attention_mask
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # 创建标签
        labels = input_ids.clone()

        # 随机掩码
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # 获取 special_tokens_mask
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True),
            dtype=torch.bool
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算掩码位置的损失

        # 80% 的时间替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% 的时间替换为随机 token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 模型参数配置类
class ModelConfig:
    def __init__(self):
        self.hidden_size = 128  # 隐藏层大小
        self.num_hidden_layers = 2  # 隐藏层数量
        self.num_attention_heads = 2  # 注意力头数量
        self.intermediate_size = 256  # 中间层大小
        self.learning_rate = 1e-5  # 学习率
        self.max_length = 512  # 最大序列长度
        self.mask_prob = 0.15  # 掩码概率

# 主程序
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a Chinese BERT model from scratch.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and tokenizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--resume_from_epoch", type=int, default=-1, help="Resume training from a specific epoch. Set to -1 to start from scratch.")
    args = parser.parse_args()

    # 初始化模型参数
    model_config = ModelConfig()

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的中文分词器
    #tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    cache_dir = "./my_cache"
    vocab_file = os.path.join(cache_dir, "vocab.txt")
    if not os.path.exists(vocab_file):
        print("本地vocab文件不存在，将从 Hugging Face 下载...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=cache_dir)
    else:
        print("本地vocab文件已存在，直接加载...")
        tokenizer = BertTokenizer.from_pretrained(cache_dir)

    # 从文件中加载文本数据
    with open(args.data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    # 创建数据集和数据加载器
    dataset = ChineseTextDataset(
        texts, tokenizer, max_length=model_config.max_length, mask_prob=model_config.mask_prob
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
    )
    model = BertForMaskedLM(config)

    # 多卡训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)

    # 检查是否从特定 epoch 继续训练
    if args.resume_from_epoch >= 0:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{args.resume_from_epoch}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resuming training from epoch {args.resume_from_epoch + 1}...")
        else:
            print(f"No checkpoint found for epoch {args.resume_from_epoch}. Starting training from scratch.")
            args.resume_from_epoch = -1

    start_epoch = args.resume_from_epoch + 1

    # 训练模型
    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss
            #print(loss.shape)  # 检查损失的形状
            #print(loss)        # 打印损失值
            #if loss.numel() > 1:
            #    loss = loss.mean()
            loss = outputs.loss.mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(dataloader)}")

        # 每个 epoch 保存一次模型和优化器状态
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        # 保存模型和分词器
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save model and tokenizer to {save_dir}")
        # 检查是否使用了 DataParallel
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(save_dir)  # 获取原始模型再保存
        else:
            model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)