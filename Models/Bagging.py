from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# 尝试用集成算法
def bagging_classsifier(x_train, y_train,x_test):
    dt = DecisionTreeClassifier()   # 创建基模型
    # lr = LogisticRegression()
    bag = BaggingClassifier(estimator=dt, n_estimators=10, random_state=42) # 创建Bagging分类器
    bag.fit(x_train, y_train)
    y_pred = bag.predict(x_test)
    return y_pred

