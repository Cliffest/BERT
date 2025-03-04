import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


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
        input_ids = inputs["input_ids"].squeeze(0)  # 每个位置上的单词 ID
        attention_mask = inputs["attention_mask"].squeeze(0)

        # 创建标签
        labels = input_ids.clone()  # [batch_size, max_length]

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



def main(args):
    # 初始化模型参数
    model_config = ModelConfig()

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的中文分词器
    #tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    cache_dir = "./my_cache"
    # 下载后需要手动改一次vocab_dir
    vocab_dir = os.path.join(cache_dir, "models--bert-base-chinese", "snapshots", "c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
    vocab_file = os.path.join(vocab_dir, "vocab.txt")
    if not os.path.exists(vocab_file):
        print("本地vocab文件不存在，将从 Hugging Face 下载...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=cache_dir)
    else:
        print("本地vocab文件已存在，直接加载...")
        tokenizer = BertTokenizer.from_pretrained(vocab_dir)

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
    assert args.resume_from_epoch >= -1
    if args.resume_from_epoch >= 0:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{args.resume_from_epoch}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resuming training from epoch {args.resume_from_epoch}...")
        else:
            #print(f"No checkpoint found for epoch {args.resume_from_epoch}. Starting training from scratch.")
            #args.resume_from_epoch = -1
            print(f"No checkpoint found for epoch {args.resume_from_epoch}.")
            assert False

    start_epoch = args.resume_from_epoch + 1

    # 训练模型
    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        total_predictions = 0
        correct_predictions = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()
            logits = outputs.logits  # 获取模型的预测结果 [batch_size, max_length, vocab_size]

            # 计算准确率
            masked_indices = labels != -100  # 找到被掩盖的位置
            _, predicted = torch.max(logits, dim=2)  # 获取预测的类别 predicted [batch_size, max_length]
            correct_predictions += (predicted == labels).masked_fill(~masked_indices, 0).sum().item()
            total_predictions += masked_indices.sum().item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f"Epoch {epoch}/{args.epochs - 1}, Loss: {total_loss / len(dataloader)}")
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch}/{args.epochs - 1}, Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy:.4f}")

        # 每若干个 epoch 保存一次模型和优化器状态
        if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            print(f"Save checkpoint to {checkpoint_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)

    # 保存模型和分词器
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, f"epoch_{args.epochs - 1}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save model and tokenizer to {save_dir}")
    # 检查是否使用了 DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(save_dir)  # 获取原始模型再保存
    else:
        model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Chinese BERT model from scratch.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and tokenizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--save_interval", type=int, default=1, help="Every how many epochs should a checkpoint be saved.")
    parser.add_argument("--resume_from_epoch", type=int, default=-1, help="Resume training from a specific epoch. Set to -1 to start from scratch.")
    args = parser.parse_args()
    main(args)