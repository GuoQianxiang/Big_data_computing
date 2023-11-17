import numpy as np
from scipy.stats import mode

class KNN():
    def __init__(self, k, ord=2):   # 创建新对象实例初始化
        self.k = k
        self.ord = ord

    def fit(self, x_train, y_train):
        # 传入参数赋值到类当中
        self.x_train = x_train
        self.y_train = y_train

        # 涉及到的操作都是numpy类型
        if type(self.x_train) != np.ndarray:
            self.x_train = np.array(self.x_train)
        if type(self.y_train) != np.ndarray:
            self.y_train = np.array(self.y_train)
        if len(self.y_train.shape) == 1:  # 维度进行延伸
            self.y_train = np.expand_dims(self.y_train, axis=1)

    # 计算的是data2中每一行数据与data1所有数据之间的各种距离，且用distances保存
    # data2：k行m列
    # data1：n行m列
    # distances：k行n列
    def calculate_distances(self, dataset1, dataset2, ord):  # 默认使用欧式距离
        distances = np.zeros((len(dataset2), len(dataset1)))  # 初始化距离矩阵
        for i in range(len(dataset2)):
            distances[i] = np.linalg.norm(dataset1 - dataset2[i], axis=1, ord=ord)  # 沿着行计算两个数组之间的差距
        return distances

    def predict(self, x_test):
        if type(x_test) != np.ndarray:
            x_test = np.array(x_test)

        distances = self.calculate_distances(self.x_train, x_test, self.ord)  # 获取距离矩阵
        partitioned_indexes = np.argpartition(distances, self.k, axis=1)[:, :self.k]  # 找到每行最小的前k个值的索引位置
        # 即获取距离最近得前k个样本
        sub_labels = self.y_train[partitioned_indexes, -1]  # 获取选定得最近前k个样本对应的标签
        modes, counts = mode(sub_labels, axis=1, keepdims=False)  # 获取每行的众数以及出现次数
        return modes  # 即为预测数