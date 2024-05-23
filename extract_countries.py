import pandas as pd
import re

# 读取数据
data = pd.read_csv('turkey_earthquake_tweets.csv')  # 请替换为你的CSV文件路径

# 创建国家名称和经纬度对照的字典
country_coords = {
    "India": (20.5937, 78.9629),
    "United States": (37.0902, -95.7129),
    "Turkey": (38.9637, 35.2433),
    "Canada": (56.1304, -106.3468),
    "Australia": (-25.2744, 133.7751),
    "United Kingdom": (55.3781, -3.4360),
    "France": (46.6034, 1.8883),
    "Germany": (51.1657, 10.4515),
    "Italy": (41.8719, 12.5674),
    "Spain": (40.4637, -3.7492),
    "China": (35.8617, 104.1954),
    "Japan": (36.2048, 138.2529),
    "South Korea": (35.9078, 127.7669),
    "Brazil": (-14.2350, -51.9253),
    "Mexico": (23.6345, -102.5528),
    "Russia": (61.5240, 105.3188),
    "South Africa": (-30.5595, 22.9375),
    # 可以继续添加其他国家
}

# 提取国家名称的函数
def extract_country(location):
    # 匹配城市, 国家格式的数据
    match = re.search(r',\s*(\w+)$', location)
    if match and match.group(1) in country_coords:
        return match.group(1)
    elif location in country_coords:
        return location
    else:
        return None

# 应用提取函数
data['Country'] = data['user_location'].apply(lambda x: extract_country(str(x)))

# 过滤掉没有匹配到国家的行
filtered_data = data.dropna(subset=['Country'])

# 根据国家名称获取经纬度
filtered_data = filtered_data.copy()
filtered_data.loc[:, 'Latitude'] = filtered_data['Country'].apply(lambda x: country_coords[x][0])
filtered_data.loc[:, 'Longitude'] = filtered_data['Country'].apply(lambda x: country_coords[x][1])

# 保存处理后的数据
filtered_data.to_csv('processed_data.csv', index=False)
