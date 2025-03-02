import re

def split_text_to_sentences(input_file, output_file):
    """
    将中文小说文本转换为每行一个句子的格式，并去掉每个句子开头的空格。
    特别处理标点符号后面紧跟引号的情况，以及省略号的特殊情况。
    
    参数:
        input_file (str): 输入文件路径（原始中文小说文本）。
        output_file (str): 输出文件路径（每行一个句子的格式）。
    """
    # 打开输入文件并读取内容
    with open(input_file, "r", encoding="utf-8") as infile:
        text = infile.read()

    # 定义处理逻辑
    def handle_punctuation(match):
        # 获取匹配的标点符号和引号
        punctuation = match.group(1)
        quotes = match.group(2) if match.group(2) else ""

        # 判断引号类型并处理
        if quotes == "”" or quotes == "’”":  # 如果是右双引号或单引号+双引号
            return f"{punctuation}{quotes}\n"
        elif quotes == "’" or quotes == "”’":  # 如果是右单引号或右双引号后紧跟右单引号
            return f"{punctuation}{quotes}\n"
        elif punctuation != "……":  # 如果不是省略号且没有引号
            return f"{punctuation}\n"
        else:  # 如果是省略号，不换行
            return f"{punctuation}"

    # 使用正则表达式匹配标点符号后面紧跟引号的情况
    # 匹配的标点符号包括：句号、问号、感叹号、分号、省略号
    # 匹配的引号包括：右双引号、单引号+双引号、右单引号、右双引号后紧跟右单引号
    text = re.sub(r"([。？！；]|)(”|’”|”’|’)?", handle_punctuation, text)

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in text.splitlines():
            # 去掉句子开头的空格
            line = line.lstrip()
            # 去掉空行
            if line.strip():
                outfile.write(line.strip() + "\n")

    print(f"处理完成！输出文件已保存到：{output_file}")

def insert_space_between_characters(input_file, output_file):
    """
    在文本文件的每一行中，每个字符（包括标点符号）之间插入一个空格，
    但不在行首和行尾插入空格。
    
    参数:
        input_file (str): 输入文件路径（原始小说文本）。
        output_file (str): 输出文件路径（每个字符之间有空格的文本）。
    """
    # 打开输入文件并读取内容
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # 按行处理，插入空格
    processed_lines = []
    for line in lines:
        line = line.strip()  # 去掉行首和行尾的空格
        if len(line) > 1:
            # 在每个字符之间插入空格，但不在行首和行尾
            spaced_line = " ".join(line)
        else:
            # 如果行只有一个字符，直接保留
            spaced_line = line
        processed_lines.append(spaced_line)

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in processed_lines:
            outfile.write(line + "\n")

    print(f"处理完成！输出文件已保存到：{output_file}")



# -> 每行一个句子
input_file = "data/金庸-倚天屠龙记txt精校版_utf-8.txt"  # 替换为你的输入文件路径
output_file = "data/倚天屠龙记.txt"  # 替换为你希望的输出文件路径
#split_text_to_sentences(input_file, output_file)

# 加空格
input_file = "data/倚天屠龙记_train_no-space.txt"
output_file = "data/倚天屠龙记_train.txt"
insert_space_between_characters(input_file, output_file)

input_file = "data/倚天屠龙记_dev_no-space.txt"
output_file = "data/倚天屠龙记_dev.txt"
insert_space_between_characters(input_file, output_file)