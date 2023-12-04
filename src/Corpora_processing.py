import re
import jieba

def process_chinese_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 使用正则表达式提取中文字符
                chinese_characters = re.findall(r'[\u4e00-\u9fa5]+', line)

                # 将提取的中文字符拼接成字符串
                processed_line = ''.join(chinese_characters)
                seg_list = jieba.cut(processed_line, cut_all=False)
                processed_line = " ".join(seg_list)

                # 将处理后的行写入输出文件
                outfile.write(processed_line + '\n')

# 示例用法
input_file_path = 'corpus/file.txt'
output_file_path = 'corpus/file_processed.txt'
process_chinese_file(input_file_path, output_file_path)

