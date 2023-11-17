import numpy as np
# 将处理id外的各行转换为numpy
def drop_id(data):
    data = data.iloc[:, 0:].dropna().values
    return data

# 对问号进行列平均值填充
def fill_question(data):
    data[data == '?'] = np.nan    # 将问号转换为 NaN 值（缺失值）
    col_mean = np.nanmean(data.astype(float), axis=0)   # 计算每列的平均值
    data[np.isnan(data.astype(float))] = np.take(col_mean, np.isnan(data.astype(float)).nonzero()[1])    # 使用列平均值填充缺失值

    # 转换数组的数据类型为 float方便后续保存
    data = data.astype(float)
    return data

def zscore(data):
    # 计算每列的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 使用 Z-Score 正交化
    normalized_data = (data - mean) / std
    print("这是正交化后的数据", normalized_data[0])
    return normalized_data

def Data_clean(data):
    data = drop_id(data)
    print("去掉id之后的数据",data[2])
    data = fill_question(data)
    print("填充问号之后的数据",data[2])
    return data
    # data = zscore(data)
    # print("zscore正交化之后的数据", data)