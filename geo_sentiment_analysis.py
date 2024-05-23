import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN, KMeans
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from geopy.distance import geodesic

# 加载数据
data = pd.read_csv('complete_dataset.csv')

# 加载已训练的模型
model = load_model('bi_lstm_model.h5')

# 定义一个函数将字符串向量转换为数值列表
def parse_vector(vector_str):
    vector_str = vector_str.strip("[]")
    vector = np.fromstring(vector_str, sep=' ')
    return vector

# 处理向量
data['text_vector'] = data['text_vector'].apply(parse_vector)
X = np.array([x for x in data['text_vector']])
X = X.reshape((X.shape[0], 1, X.shape[1]))  # 修改输入形状

# 预测情感标签
y_pred = model.predict(X)
data['sentiment_label'] = np.argmax(y_pred, axis=1)

# 将情感标签转换为文本
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
data['sentiment'] = data['sentiment_label'].map(label_mapping)

# 转换日期格式
data['date'] = pd.to_datetime(data['date'])

# 时间序列分析
def plot_time_series(data):
    plt.figure(figsize=(12, 6))
    data.set_index('date', inplace=True)
    sentiment_counts = data.resample('D').sentiment.value_counts().unstack().fillna(0)
    sentiment_counts.plot(kind='line', stacked=True, colormap='coolwarm')
    plt.title('Sentiment Analysis Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.show()

plot_time_series(data.copy())

# 地理热图绘制
def plot_geographical_heatmap(data):
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    base_map = folium.Map(location=map_center, zoom_start=5)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in data.iterrows()]
    HeatMap(heat_data).add_to(base_map)
    return base_map

geo_heatmap = plot_geographical_heatmap(data)
geo_heatmap.save('geo_heatmap.html')

# 空间聚类分析
def spatial_clustering(data):
    coords = data[['Latitude', 'Longitude']].values
    db = DBSCAN(eps=0.5, min_samples=10).fit(coords)
    labels = db.labels_
    data['cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Longitude', y='Latitude', hue='cluster', palette='viridis', data=data)
    plt.title('DBSCAN Spatial Clustering')
    plt.show()

spatial_clustering(data.copy())

# 时间聚类分析
def temporal_clustering(data):
    data['day_of_year'] = data['date'].dt.dayofyear
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data[['day_of_year']])
    data['time_cluster'] = kmeans.labels_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='day_of_year', y='sentiment_label', hue='time_cluster', palette='coolwarm', data=data)
    plt.title('KMeans Temporal Clustering')
    plt.xlabel('Day of Year')
    plt.ylabel('Sentiment Label')
    plt.show()
    return data

data = temporal_clustering(data.copy())

# 结合分析
def combined_analysis(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Longitude', y='Latitude', hue='sentiment', style='time_cluster', data=data, palette='coolwarm')
    plt.title('Geospatial and Temporal Sentiment Analysis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Sentiment and Time Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

combined_analysis(data.copy())

# 按地理位置分析情感分布
def analyze_sentiment_by_distance(data, ref_lat, ref_long):
    # 计算每个推文到震中（ref_lat, ref_long）的距离
    data['distance'] = data.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), (ref_lat, ref_long)).km, axis=1)

    # 按距离分组并计算情感分布
    distance_bins = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    data['distance_group'] = pd.cut(data['distance'], bins=distance_bins)
    sentiment_distribution = data.groupby('distance_group')['sentiment'].value_counts(normalize=True).unstack().fillna(0)

    # 绘制情感分布图
    sentiment_distribution.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(12, 6))
    plt.title('Sentiment Distribution by Distance from Epicenter')
    plt.xlabel('Distance from Epicenter (km)')
    plt.ylabel('Proportion of Sentiments')
    plt.legend(title='Sentiment')
    plt.show()

# 设定震中的经纬度（以土耳其为例）
epicenter_lat = 37.174
epicenter_long = 37.032
analyze_sentiment_by_distance(data.copy(), epicenter_lat, epicenter_long)
