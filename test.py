import torch
from transformers import BertTokenizer, BertForMaskedLM


# 1. 加载训练好的模型和 tokenizer
n_epoch = 1
model_path = f"bert_novel_model/epoch_{n_epoch}"  # 训练时保存的模型路径
tokenizer_path = model_path + "/vocab.txt"  # 训练时保存的 tokenizer 词汇表

# tokenizer = BertTokenizer(vocab_file=tokenizer_path)
# model = BertForMaskedLM.from_pretrained(model_path)

# Pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)
model.eval()

# 2. 测试句子
test_sentence = "他说到这里，伸[MASK]贴在韦一笑后心灵台穴上，运气助他抵御寒毒。"

# 3. 进行分词
inputs = tokenizer(test_sentence, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# 4. 预测 `[MASK]` 位置的单词
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# 找到 [MASK] 位置
mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

# 取 Top 5 预测结果
top_5_tokens = torch.topk(predictions[0, mask_index], 5).indices.tolist()

# 5. 输出预测结果
print(f"输入句子: {test_sentence}")
print("BERT 预测的可能词语:")
for token in top_5_tokens:
    print(f"- {tokenizer.decode([token])}")
