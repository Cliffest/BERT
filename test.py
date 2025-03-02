import argparse
from transformers import BertTokenizerFast, BertForMaskedLM
import torch


parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_epoch', type=int, required=True, help="-1 if testing with internet model")
parser.add_argument('--mask_token_ids', nargs='+', type=int, required=True, help="")
args = parser.parse_args()

test_text = "张无忌心想，自已如此落魄，倘若提起太师父和父母的名字，当真辱没了他们。"

# 加载模型和分词器
if args.n_epoch >= 0:  # Ours
    model_path = "outputs/" + f"epoch_{args.n_epoch}"  # 替换为你的模型保存路径
    print(f"Load from {model_path}")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
else:  # Pretrained model
    print("Load from Hugging Face")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 准备测试文本
inputs = tokenizer(test_text, return_tensors="pt")

# 替换多个词为 [MASK]
# 假设我们想预测第5个和第7个词（从0开始计数）
mask_token_indices = args.mask_token_ids  # 需要掩码的位置
for idx in mask_token_indices:
    inputs["input_ids"][0, idx] = tokenizer.mask_token_id

# 打印掩码后的句子
masked_sentence = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
#print(f"掩码后的句子: {masked_sentence}")

# 预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 获取每个掩码位置的前5个预测结果
predicted_tokens = []  # 用于存储每个掩码位置的预测单词
for idx in mask_token_indices:
    # 获取当前掩码位置的预测结果
    logits = predictions[0, idx]
    top_5_indices = torch.topk(logits, k=5).indices
    top_5_tokens = tokenizer.convert_ids_to_tokens(top_5_indices)
    top_5_probabilities = torch.softmax(logits, dim=0)[top_5_indices].tolist()

    print(f"\n位置 {idx} 的预测结果：")
    for token, prob in zip(top_5_tokens, top_5_probabilities):
        print(f"  Token: {token}, 概率: {prob:.4f}")
    
    # 选择概率最高的预测单词
    predicted_tokens.append(top_5_tokens[0])

# 替换回预测的单词并打印完整句子
predicted_input_ids = inputs["input_ids"].clone()  # 创建一个副本
for idx, token in zip(mask_token_indices, predicted_tokens):
    predicted_input_ids[0, idx] = tokenizer.convert_tokens_to_ids(token)
predicted_sentence = tokenizer.decode(predicted_input_ids[0], skip_special_tokens=True)

print(f"\n原始句子: {test_text}")
print(f"掩码后的句子: {masked_sentence}")
print(f"预测后的句子: {predicted_sentence}")