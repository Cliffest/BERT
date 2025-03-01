def check_characters(input_files, dictionary_file, output_file):
    """
    检查多个输入文件中的所有字符是否在字典文件中存在。
    如果某个字符不存在，则将其写入输出文件。
    
    参数:
        input_files (list): 输入文件路径列表（小说文本）。
        dictionary_file (str): 字典文件路径（包含所有允许的字符）。
        output_file (str): 输出文件路径（保存不存在的字符）。
    """
    # 读取字典文件中的所有字符
    with open(dictionary_file, "r", encoding="utf-8") as dict_file:
        dictionary_chars = set(dict_file.read())

    # 读取所有输入文件中的字符
    all_input_chars = set()
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as infile:
            all_input_chars.update(infile.read())

    # 找出在字典中不存在的字符
    missing_chars = all_input_chars - dictionary_chars

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        for char in missing_chars:
            if char == " ":
                continue
            outfile.write(char + "\n")
            outfile.write("##" + char + "\n")

    print(f"检查完成！不存在的字符已保存到：{output_file}")



# 使用示例
input_files = ["data/倚天屠龙记_train.txt", "data/倚天屠龙记_dev.txt"]  # 替换为你的输入文件路径列表
dictionary_file = "data/bert_vocab.txt"  # 替换为你的字典文件路径
output_file = "data/Add_to_vocab.txt"  # 替换为你希望的输出文件路径
check_characters(input_files, dictionary_file, output_file)