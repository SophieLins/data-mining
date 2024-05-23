import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 加载预训练的Word2Vec模型
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 读取数据
data = pd.read_csv('final_data.csv')

# 清洗文本
stop_words = set(stopwords.words('english'))
def clean_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered_words)

data['cleaned_text'] = data['cleaned_text'].apply(clean_text)

# 将文本转换为向量
def text_to_vector(text):
    words = text.split()
    vector = np.mean([model[word] for word in words if word in model], axis=0)
    return vector if isinstance(vector, np.ndarray) else np.zeros(300)

data['text_vector'] = data['cleaned_text'].apply(text_to_vector)

# 情感分析
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
data['sentiments'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
data['sentiment_label'] = data['sentiments'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

# 保存完整数据集
data.to_csv('complete_dataset.csv', index=False)

# 划分数据集
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 保存数据集
train.to_csv('train_dataset.csv', index=False)
test.to_csv('test_dataset.csv', index=False)
