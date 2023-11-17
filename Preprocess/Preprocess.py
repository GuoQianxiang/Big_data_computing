import numpy as np
import chardet
import pandas as pd
from Codes.Big_Data_Computing.Data2_Code.Preprocess.data_clean import Data_clean

# 读入csv文件
def read_files(filename1, filename2):
    training = pd.read_csv(filename1)
    validation = pd.read_csv(filename2)
    print("读取的数据pandas",training)
    return training,validation

def preprocess(data1,data2):
    data1, data2 = read_files(filename1, filename2)
    data1 = Data_clean(data1)
    data2 = Data_clean(data2)
    return data1, data2

if __name__ == "__main__":
    filename1 = 'training.csv'
    filename2 = 'validation.csv'
    data1, data2 = preprocess(filename1, filename2)

    # 使用 numpy.savetxt() 将数据保存为 CSV 文件
    np.savetxt('cleaned_training_data', data1, delimiter=',')
    np.savetxt('cleaned_validation_data', data2, delimiter=',')

    print('清洗后的文件保存成功。')