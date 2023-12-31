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

        # 初始化嵌入层权重
        self.in_embed.weight.data.uniform_(-0.5 / embed_size, 0.5 / embed_size)
        self.out_embed.weight.data.uniform_(-0.5 / embed_size, 0.5 / embed_size)

    # 前向传播，计算目标词和上下文词之间的得分
    def forward(self, target, context):
        in_embeds = self.in_embed(target)
        out_embeds = self.out_embed(context)

        # 计算得分
        scores = torch.matmul(in_embeds, out_embeds.t())

        return scores

    # Negative Sampling 损失计算
    def negative_sampling_loss(self, target, context, neg_samples):
        in_embeds = self.in_embed(target)
        out_embeds = self.out_embed(context)

        # 计算正样本得分
        positive_scores = torch.sum(in_embeds * out_embeds, dim=1).squeeze().sigmoid()

        # 计算负样本得分
        neg_embeds = self.out_embed(neg_samples)
        negative_scores = torch.sum(-in_embeds * neg_embeds, dim=2).squeeze().sigmoid()

        # 计算 Negative Log-Likelihood Loss
        loss = -torch.log(positive_scores) - torch.sum(torch.log(negative_scores))
        return loss.mean()

# # 示例用法
# vocab_size = 10000
# embed_size = 300
# model = SkipGramModel(vocab_size, embed_size)

# # 假设 target, context, neg_samples 是合适的输入
# loss = model.negative_sampling_loss(target, context, neg_samples)


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

    
    # 论文中使用 Adagrad 优化器，学习率为 0.01
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)


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


# 示例用法
text_data = ["the", "quick", "brown", "fox",
             "jumps", "over", "the", "lazy", "dog"]
model, dataset = train_word2vec_model(text_data)

# 获取词向量
word_idx = dataset.word_to_index["fox"]
word_vector = model.in_embed(torch.tensor(word_idx)).detach().numpy()

print(f"Vector for 'fox': {word_vector}")
