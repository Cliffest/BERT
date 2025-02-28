import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Fine-tuning: 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 读取小说文本
txt_path = "data/金庸-倚天屠龙记txt精校版_utf-8.txt"
print(f"Loading from {txt_path}")
with open(txt_path, "r", encoding="utf-8") as file:
    text = file.read()

# 将文本切分为小段落或句子
text_lines = text.splitlines()

# 自定义 Dataset 类
class NovelDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 为 Masked Language Modeling 创建标签
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 将文本数据集分割为训练集和验证集
train_texts, eval_texts = train_test_split(text_lines, test_size=0.1)

# 创建训练集和验证集的数据集对象
train_dataset = NovelDataset(train_texts, tokenizer)
eval_dataset = NovelDataset(eval_texts, tokenizer)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

# 设置优化器和损失函数
optimizer = AdamW(model.parameters(), lr=5e-5)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)

# 定义训练过程
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        optimizer.zero_grad()
        
        # 将数据移动到 GPU（如果可用）
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 进行前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 进行反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 定义评估过程
def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 进行前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(eval_loader)

# 训练模型
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 训练阶段
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    # 验证阶段
    eval_loss = evaluate(model, eval_loader, device)
    print(f"Validation Loss: {eval_loss:.4f}")
    
    # 保存模型
    model.save_pretrained(f"./bert_novel_model/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"./bert_novel_model/epoch_{epoch+1}")
