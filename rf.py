import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import ast

# 定义一个函数将字符串向量转换为数值列表
def parse_vector(vector_str):
    vector_str = vector_str.strip("[]")
    vector = np.fromstring(vector_str, sep=' ')
    return vector

# 加载数据
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# 处理向量
X_train = np.vstack(train_data['text_vector'].apply(parse_vector))
X_test = np.vstack(test_data['text_vector'].apply(parse_vector))

# 标签编码
encoder = LabelEncoder()
encoder.fit(train_data['sentiment_label'])
y_train = encoder.transform(train_data['sentiment_label'])
y_test = encoder.transform(test_data['sentiment_label'])

# 构建随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 计算混淆矩阵和分类报告
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

report = classification_report(y_test, y_pred, target_names=encoder.classes_)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm_df)

print("\nClassification Report:")
print(report)

