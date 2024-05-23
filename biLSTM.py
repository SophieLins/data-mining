import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# 定义一个函数将字符串向量转换为数值列表
def parse_vector(vector_str):
    vector_str = vector_str.strip("[]")
    vector = np.fromstring(vector_str, sep=' ')
    return vector

# 加载数据
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# 处理向量
X_train = np.array([x.reshape(1, -1) for x in train_data['text_vector'].apply(parse_vector)])
X_test = np.array([x.reshape(1, -1) for x in test_data['text_vector'].apply(parse_vector)])

# 标签编码
encoder = LabelEncoder()
encoder.fit(train_data['sentiment_label'])
y_train = encoder.transform(train_data['sentiment_label'])
y_train = to_categorical(y_train)
y_test = encoder.transform(test_data['sentiment_label'])
y_test = to_categorical(y_test)

# 构建模型
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, 300)))  # 注意这里的input_shape
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 保存模型
model.save('bi_lstm_model.h5')

# 加载模型
model = load_model('bi_lstm_model.h5')

# 预测测试集
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 转换预测结果为类别标签
y_true = np.argmax(y_test, axis=1)  # 真实的类别标签

# 计算混淆矩阵和分类报告
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

report = classification_report(y_true, y_pred_classes, target_names=encoder.classes_)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm_df)

print("\nClassification Report:")
print(report)
