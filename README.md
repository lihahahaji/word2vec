# 实验

## 1 gensim 实现 word2vec

[gensim官方文档](https://radimrehurek.com/gensim/models/word2vec.html)

使用Gensim训练Word2vec：

- 安装 Gensim 库

```
pip3 install gensim
```

#### gensim - word2vec 模型参数

1. **sentences (list of list of str):** 输入语料库，每个元素是一个句子，每个句子是一个单词列表。

2. **sg (int, optional, default=0):** 选择使用哪种模型。sg=0表示使用Skip-gram模型，sg=1表示使用CBOW模型。

3. **size (int, optional, default=100):** 词向量的维度，即每个词的向量表示的维数。

4. **window (int, optional, default=5):** 上下文窗口的大小，表示在训练过程中考虑目标词周围的词的范围。

5. **min_count (int, optional, default=5):** 忽略出现次数低于该值的单词。

6. **workers (int, optional, default=3):** 训练的并行度，表示使用多少个CPU核心来训练模型。

7. **sg (int, optional, default=0):** 选择使用哪种模型。sg=0表示使用Skip-gram模型，sg=1表示使用CBOW模型。

8. **hs (int or {0, 1}, optional, default=0):** 选择使用softmax（hs=0）还是负采样（hs=1）。负采样通常在小规模数据集上更快。

9. **negative (int, optional, default=5):** 对于负采样，设置多少个噪声词作为负样本。

10. **ns_exponent (float, optional, default=0.75):** 负采样的指数，通常取值在[0.5, 1.0]之间。

11. **alpha (float, optional, default=0.025):** 初始学习率。

12. **min_alpha (float, optional, default=0.0001):** 学习率的下限。学习率会在训练过程中逐渐减小。

13. **iter (int, optional, default=5):** 迭代次数，即训练数据的多少遍。



使用示例：

```python
from gensim.models import Word2Vec

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# 保存模型
model.save("word2vec_model.model")

# 加载模型
model = Word2Vec.load("word2vec_model.model")
```



#### 训练步骤

1. 语料库预处理:

   一行一个文档或句子，将文档或句子分词以空格分割（英文可以不用分词，英文单词之间已经由空格分割，中文预料需要使用分词工具进行分词，常见的分词工具有StandNLP、ICTCLAS、Ansj、FudanNLP、HanLP、结巴分词等）。

   - 处理文本信息

   - 使用 jieba 分词

     ```python
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
     
     
     ```

     

2. 将原始的训练语料转化成一个sentence的迭代器，每一次迭代返回的sentence是一个word（utf8格式）的列表。可以使用Gensim中word2vec.py中的`LineSentence()`方法实现；

   ```python
   # 使用 LineSentence 读取文本数据
   sentences = LineSentence(data_file)
   ```

3. 将上面处理的结果输入Gensim内建的word2vec对象进行训练

4. 默认用了CBOW模型，采用高频词抽样+负采样进行优化



#### 训练代码：

```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 替换为你的数据文件路径
data_file = 'corpus/file_processed.txt'

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
model.save('./params/model.bin')

# 加载模型
# model = Word2Vec.load('path/to/save/model.bin')

# 获取词向量

vector = model.wv['地球']

# 查找与给定词最相似的词汇
similar_words = model.wv.most_similar('地球', topn=5)

# 打印结果
print(f"Vector for '地球': {vector}")
print(f"Most similar words to '地球': {similar_words}")

```



#### 实验结果:

![image-20231204112011207](./assets/论文复现.assets/image-20231204112011207.png)





## 2 Pytorch 实现 word2vec

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义数据集类，处理Word2Vec模型的训练数据


class Word2VecDataset(Dataset):
    # 初始化数据集
    def __init__(self, text_data):
        self.text_data = text_data
        # 将text_data中的词语转换为集合（set），去除重复的词语，然后再转换为列表（list）
        self.vocab = list(set(self.text_data))
        # 创建一个从词语到索引的映射，每个词语被映射为一个索引
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        # 创建一个从索引到词语的映射，每个索引对应于self.vocab中的一个词语
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    # 返回数据集的大小
    def __len__(self):
        return len(self.text_data)

    # 返回目标词和上下文词
    def __getitem__(self, idx):
        target_word = self.word_to_index[self.text_data[idx]]
        context_words = [self.word_to_index[word]
                         for word in self.get_context(idx)]
        return target_word, context_words

    # 获取上下文词
    def get_context(self, idx, window_size=2):
        start = max(0, idx - window_size)
        end = min(len(self.text_data), idx + window_size + 1)
        context = [self.text_data[i] for i in range(start, end) if i != idx]
        return context

# 定义Skip-gram模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        # 两个嵌入层，分别用于表示输入词和输出词
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    # 前向传播，计算目标词和上下文词之间的得分
    def forward(self, target, context):
        in_embeds = self.in_embed(target)
        out_embeds = self.out_embed(context)
        scores = torch.matmul(in_embeds, out_embeds.t())
        return scores

# 定义 CBOW 模型
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOWModel, self).__init__()

        # 定义输入和输出的Embedding层
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    # 前向传播，输入是上下文词的嵌入向量，输出是目标词的得分
    def forward(self, context):
        # 获取上下文词的嵌入向量并对它们进行求和
        context_embeds = self.in_embed(context)
        sum_embeds = torch.sum(context_embeds, dim=1)

        # 计算求和后的嵌入向量与目标词嵌入之间的得分
        scores = torch.matmul(sum_embeds, self.out_embed.weight.t())

        # 返回得分
        return scores


# 训练模型
def train_word2vec_model(text_data, embed_size=100, num_epochs=5, learning_rate=0.01):
    # 获取词汇表大小和创建数据集对象
    vocab_size = len(set(text_data))
    dataset = Word2VecDataset(text_data)

    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 初始化Skip-gram模型、损失函数和优化器:
    model = SkipGramModel(vocab_size, embed_size)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用随机梯度下降（SGD）优化器，学习率为 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for target, context in data_loader:
            # 清零之前的梯度，以避免梯度累积
            optimizer.zero_grad()
            # 计算模型的输出得分
            scores = model(target, context)
            # 计算模型输出与真实值之间的损失
            loss = criterion(scores.view(-1, vocab_size), context.view(-1))
            # 反向传播，计算梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()
            total_loss += loss.item()

        # 在每个epoch结束时打印平均损失
        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')
        
    # 返回训练好的模型和词汇表索引映射
    return model, dataset


```
