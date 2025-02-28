import argparse
import torch

from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments


def train():
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 读取txt文件
    print(f"Loading from {args.txt}")
    with open(args.txt, "r", encoding="utf-8") as file:
        text = file.read()

    # 将文本转换为数据集
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # 创建Dataset对象
    dataset = Dataset.from_dict({"text": [text]})  # 将整个文本作为单个样本

    # 使用BERT的分词器对文本进行分词
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 加载BERT模型
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./bert_novel_model",  # 输出目录
        overwrite_output_dir=True,  # 如果输出目录已存在，覆盖
        num_train_epochs=3,  # 训练3个epoch
        per_device_train_batch_size=8,  # 每个设备的训练批次大小
        save_steps=10_000,  # 保存模型的频率
        logging_steps=500,  # 日志记录频率
        prediction_loss_only=True,  # 只计算损失
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # 开始训练
    trainer.train()

    # 保存模型和分词器
    model.save_pretrained("./bert_novel_model")
    tokenizer.save_pretrained("./bert_novel_model")

def test():
    # 加载训练好的模型
    model = BertForMaskedLM.from_pretrained("./bert_novel_model")
    tokenizer = BertTokenizer.from_pretrained("./bert_novel_model")

    # 输入文本进行推理
    input_text = "他说到这里，伸[MASK]贴在韦一笑后心灵台穴上，运气助他抵御寒毒。"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 进行预测
    with torch.no_grad():
        logits = model(**inputs).logits

    # 获取[MASK]位置的预测
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    print(f"Predicted token: {predicted_token}")



def main(args):
    if args.train:
        train()
    
    if args.test:
        test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--txt', type=str, default="data/金庸-倚天屠龙记txt精校版_utf-8.txt", help=".txt file path")
    parser.add_argument('--train', action='store_true', help="")
    parser.add_argument('--test', action='store_true', help="")
    args = parser.parse_args()
    main(args)