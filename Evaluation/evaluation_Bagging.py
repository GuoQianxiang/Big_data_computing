from sklearn.metrics import f1_score
from Codes.Big_Data_Computing.Data2_Code.Models.Knn import KNN
from Codes.Big_Data_Computing.Data2_Code.Preprocess.data_convert import data_cut
from Codes.Big_Data_Computing.Data2_Code.Preprocess.Preprocess import read_files
from Codes.Big_Data_Computing.Data2_Code.Models.Bagging import bagging_classsifier
import os

# 评分函数——使用F1 score和准确率
def measurement(y_test, y_pred):
    #评分函数的实现
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1-score-macro: {:.2f}".format(f1))
    f1 = f1_score(y_test, y_pred, average='micro')
    print("F1-score-micro: {:.2f}".format(f1))
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("准确率为：%.2f" % accuracy)

if __name__ == "__main__":
    file_path1 = 'D:\SoftWare\PyCharm\Project\Test\Codes\Big_Data_Computing\Project\data2_code\Big_data_computing\Preprocess\cleaned_training_data'
    file_path2 = 'D:\SoftWare\PyCharm\Project\Test\Codes\Big_Data_Computing\Project\data2_code\Big_data_computing\Preprocess\cleaned_validation_data'
    training, validation = read_files(file_path1, file_path2)
    x_train, x_test, y_train, y_test = data_cut(training,validation)
    y_pred = bagging_classsifier(x_train, y_train, x_test)
    measurement(y_test,y_pred)