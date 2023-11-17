
# 将数据集的训练内容和标签切割开
def data_cut(training, validation):
    # 对引入的数据按照数据和标签进行切割
    x_train = training[:, :-1]  # 得到训练集的数据
    x_test = validation[:, :-1]  # 得到验证集的数据
    y_train = training[:, -1]  # 得到训练集的标签
    y_test = validation[:, -1]  # 得到验证集的标签

    return x_train, x_test, y_train, y_test
