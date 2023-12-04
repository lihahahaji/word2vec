import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re

# 解析 XML 文件
tree = ET.parse('./corpus/A1E.xml')
root = tree.getroot()

# 提取<wtext>标签内的文本内容
text_data = ""
for wtext_element in root.iter('wtext'):
    if wtext_element.text is not None:
        text_data += wtext_element.text + ' '

# 预处理函数
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号等特殊字符
    words = text.split()  # 分词
    return words

# 对文本进行预处理
preprocessed_text = preprocess_text(text_data)

# 将预处理后的文本保存到文件中
with open('preprocessed_corpus.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(preprocessed_text))

# 使用 LineSentence 读取预处理后的文本数据
sentences = LineSentence('preprocessed_corpus.txt')

# 定义 Word2Vec 模型参数
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# 训练模型
model.train(sentences, total_examples=model.corpus_count, epochs=10)

# 保存模型
model.save('word2vec_model.bin')

# 加载模型
# model = Word2Vec.load('word2vec_model.bin')
