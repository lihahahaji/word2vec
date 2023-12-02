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
        context_words = [self.word_to_index[word] for word in self.get_context(idx)]
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
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

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
    vocab_size = len(set(text_data))
    dataset = Word2VecDataset(text_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SkipGramModel(vocab_size, embed_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for target, context in data_loader:
            optimizer.zero_grad()
            scores = model(target, context)
            loss = criterion(scores.view(-1, vocab_size), context.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

    return model, dataset.index_to_word

# 示例用法
text_data = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
model, index_to_word = train_word2vec_model(text_data)

# 获取词向量
word_idx = dataset.word_to_index["fox"]
word_vector = model.in_embed(torch.tensor(word_idx)).detach().numpy()

print(f"Vector for 'fox': {word_vector}")
