import argparse
import os
import torch

from transformers import BertTokenizer, BertForMaskedLM


def main(args):
    test_text = "张无忌心想，自已如此落魄，倘若提起太师父和父母的名字，当真辱没了他们。"

    # 加载模型和分词器
    if args.n_epoch >= 0:  # Ours
        assert not args.model_dir == None
        model_path = os.path.join(args.model_dir, f"epoch_{args.n_epoch}")  # 替换为你的模型保存路径
        assert (os.path.exists(os.path.join(model_path, "vocab.txt")) and 
                os.path.exists(os.path.join(model_path, "model.safetensors")))
        print(f"Load from {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
    else:  # Pretrained model
        print("Load from Hugging Face")
        cache_dir = "./my_cache"
        # 下载后需要手动改一次 vocab_dir 和 model_dir
        vocab_dir = os.path.join(cache_dir, "models--bert-base-chinese", "snapshots", "c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
        model_dir = os.path.join(cache_dir, "models--bert-base-chinese", "snapshots", "c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

        vocab_file = os.path.join(vocab_dir, "vocab.txt")
        if not os.path.exists(vocab_file):
            print("本地 vocab 文件不存在，将从 Hugging Face 下载...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=cache_dir)
        else:
            print("本地 vocab 文件已存在，直接加载...")
            tokenizer = BertTokenizer.from_pretrained(vocab_dir)
        
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            print("本地 model 不存在，将从 Hugging Face 下载...")
            model = BertForMaskedLM.from_pretrained("bert-base-chinese", cache_dir=cache_dir)
        else:
            print("本地 model 已存在，直接加载...")
            model = BertForMaskedLM.from_pretrained(model_dir)

    # 准备测试文本
    inputs = tokenizer(test_text, return_tensors="pt")

    # 替换多个词为 [MASK]
    # 假设我们想预测第5个和第7个词（从0开始计数）: mask_token_indices = [5, 7]
    mask_token_indices = args.mask_token_ids  # 需要掩码的位置
    for idx in mask_token_indices:
        inputs["input_ids"][0, idx] = tokenizer.mask_token_id

    # 掩码后的句子
    masked_sentence = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

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
    predicted_sentence = predicted_sentence.replace(" ", "")

    print(f"\n原始句子: {test_text}")
    print(f"掩码后的句子: {masked_sentence}")
    print(f"预测后的句子: {predicted_sentence}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--model_dir", type=str, default=None, help="Directory to load the trained model and tokenizer.")
    parser.add_argument('--n_epoch', type=int, required=True, help="-1 if testing with internet model")
    parser.add_argument('--mask_token_ids', nargs='+', type=int, required=True, help="")
    args = parser.parse_args()
    main(args)