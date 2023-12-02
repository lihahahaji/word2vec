from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 替换为你的数据文件路径
data_file = 'path/to/your/data.txt'

# 使用 LineSentence 读取文本数据
sentences = LineSentence(data_file)

# 定义 Word2Vec 模型参数
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# vector_size: 词向量的维度
# window: 上下文窗口大小，表示当前词与预测词的最大距离
# min_count: 忽略出现次数少于min_count的词
# workers: 训练并行化的线程数

# 训练模型
model.train(sentences, total_examples=model.corpus_count, epochs=10)

# 保存模型
model.save('path/to/save/model.bin')

# 加载模型
# model = Word2Vec.load('path/to/save/model.bin')
