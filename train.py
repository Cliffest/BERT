import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from enum import Enum, auto
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler  # DDP
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


class ModelConfig:
    def __init__(self):
        self.hidden_size = 128        # 隐藏层大小
        self.num_hidden_layers = 2    # 隐藏层数量
        self.num_attention_heads = 2  # 注意力头数量
        self.intermediate_size = 256  # 中间层大小
        self.learning_rate = 1e-5     # 学习率
        self.max_length = 512         # 最大序列长度
        self.mask_prob = 0.15         # 掩码概率

class Mode(Enum):
    CPU = auto()
    one_GPU = auto()
    DP = auto()
    DDP = auto()

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


def set_mode(mode):
    assert mode in ["CPU", "1_GPU", "DP", "DDP"]
    return Mode.CPU if mode == "CPU" else (
           Mode.one_GPU if mode == "1_GPU" else (
           Mode.DP if mode == "DP" else (
           Mode.DDP if mode == "DDP" else None)))

def init_distributed_mode():  # DDP, 初始化分布式环境
    assert all(variable in os.environ for variable in ["WORLD_SIZE",    # nnodes * nproc-per-node
                                                       "RANK",          # 0~(WORLD_SIZE-1), torchrun 自动设置
                                                       "MASTER_ADDR",   # master_addr
                                                       "MASTER_PORT"])  # master_port
    # 设置当前进程的 GPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # torchrun 自动设置 LOCAL_RANK
    # 从环境变量中读取
    print(f"Initializing distributed training on local_rank {int(os.environ['LOCAL_RANK'])}, global_rank {int(os.environ['RANK'])}, world_size {int(os.environ['WORLD_SIZE'])}")
    # 初始化分布式环境
    dist.init_process_group(backend="nccl", init_method="env://")
    print("Distributed training initialized successfully.")

def load_tokenizer():
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
    return tokenizer

def get_model(model):
    return model.module if isinstance(model, DP) or isinstance(model, DDP) else model

def load_from_epoch(args, model, optimizer, device):
    assert args.resume_from_epoch >= 0
    if args.resume_from_epoch > 0:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{args.resume_from_epoch}.pth")
        print(f"Resume training from {checkpoint_path}...")
        assert os.path.exists(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not (args.mode is Mode.DDP) or dist.get_rank() == 0:  # 若采用 DDP, 只在主进程（rank 0）载入模型
            get_model(model).load_state_dict(checkpoint['model_state_dict'])  # 处理是否使用了 DP/DDP
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return args.resume_from_epoch + 1, model, optimizer

def save_checkpoint(args, epoch, model, optimizer):
    os.makedirs(args.output_dir, exist_ok=True)
    # 保存断点
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": get_model(model).state_dict(),  # 处理是否使用了 DP/DDP
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Save checkpoint to {checkpoint_path}")

def save_model(args, model, tokenizer):
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, f"epoch_{args.epochs}")
    os.makedirs(save_dir, exist_ok=True)
    get_model(model).save_pretrained(save_dir)  # 处理是否使用了 DP/DDP
    tokenizer.save_pretrained(save_dir)
    print(f"Save model and tokenizer to {save_dir}")



def main(args):
    args.mode = set_mode(args.mode)
    assert torch.cuda.is_available() if args.mode in [Mode.one_GPU, Mode.DP, Mode.DDP] else True

    # 初始化分布式环境
    if args.mode is Mode.DDP:  # DDP
        init_distributed_mode()

    # 加载预训练的中文分词器
    tokenizer = load_tokenizer()

    # 初始化模型参数
    model_config = ModelConfig()

    # 从文件中加载文本数据
    with open(args.data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    # 创建数据集和数据加载器
    dataset = ChineseTextDataset(
        texts, tokenizer, max_length=model_config.max_length, mask_prob=model_config.mask_prob
    )
    if args.mode in [Mode.CPU, Mode.one_GPU, Mode.DP]:  # 不启用分布式训练
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:  # DDP
        data_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=data_sampler)

    # 初始化模型
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
    )
    model = BertForMaskedLM(config)

    # 配置设备
    device = torch.device("cpu" if args.mode is Mode.CPU else "cuda")
    print(f"Using device: {device}")

    model.to(device)

    # 分布式训练
    if args.mode in [Mode.DP, Mode.DDP]:
        print(f"Using {torch.cuda.device_count()} GPU"+("s" if torch.cuda.device_count() > 1 else ""))
        if args.mode is Mode.DP:  # DP
            model = DP(model)
        else:  # DDP
            model = DDP(model, device_ids=[torch.cuda.current_device()])

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)

    # 检查是否从特定 epoch 继续训练
    start_epoch, model, optimizer = load_from_epoch(args, model, optimizer, device)
    
    # 训练模型
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        total_loss = 0
        total_predictions = 0
        correct_predictions = 0

        if args.mode is Mode.DDP:
            data_sampler.set_epoch(epoch)  # DDP

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

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
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        accuracy = correct_predictions / total_predictions
        print(f"    Loss: {total_loss / len(dataloader)}, Accuracy: {(100*accuracy):.2f} %")

        if not (args.mode is Mode.DDP) or dist.get_rank() == 0:  # 若采用 DDP, 只在主进程（rank 0）保存模型
            # 每若干个 epoch 保存一次模型和优化器状态
            if epoch == 1 or epoch % args.save_interval == 0 or epoch == args.epochs:
                save_checkpoint(args, epoch, model, optimizer)

    if not (args.mode is Mode.DDP) or dist.get_rank() == 0:  # 若采用 DDP, 只在主进程（rank 0）保存模型
        # 保存模型和分词器
        save_model(args, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Chinese BERT model from scratch.")
    parser.add_argument('--mode', type=str, default="1_GPU", help="Mode in [CPU, 1_GPU, DP, DDP].")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and tokenizer.")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--save_interval", type=int, default=1, help="Every how many epochs should a checkpoint be saved.")
    parser.add_argument("--resume_from_epoch", type=int, default=0, help="Resume training from a specific epoch. Set to 0 to start from scratch.")
    args = parser.parse_args()
    main(args)