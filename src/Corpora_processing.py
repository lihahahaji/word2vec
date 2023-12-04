import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_corpus(corpus):
    processed_corpus = []
    
    # 分词
    for document in corpus:
        tokens = word_tokenize(document)
        
        # 小写化
        tokens = [word.lower() for word in tokens]
        
        # 去除停用词和标点符号
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        processed_corpus.append(tokens)
    
    return processed_corpus

# 示例语料库
corpus = [
    "This is a sample sentence.",
    "And here's another one.",
    "Let's preprocess this text data!"
]

# 预处理语料库
preprocessed_corpus = preprocess_corpus(corpus)

# 打印预处理后的语料库
for i, document in enumerate(preprocessed_corpus):
    print(f"Document {i + 1}: {document}")
