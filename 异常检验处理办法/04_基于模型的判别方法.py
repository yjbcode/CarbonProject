"""
基于模型的判别法
· 思路：用模型预测值（y_pred）与真实值（y）的偏差衡量异常点
· 标准：计算(y-y_pred)的z值，其绝对值大于n倍σ的点视作异常点
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
#%matplotlib inline

# 数据
x = datasets.load_boston().data
y = datasets.load_boston().target

# 建模（本例采用线性回归模型）
model = LinearRegression()
model.fit(x, y)
y_pred = pd.Series(model.predict(x))  # 计算预测值

# 判别依据
resid = y - y_pred        # 预测值与真实值的偏差
meanResid = resid.mean()  # 偏差的均值
stdResid = resid.std()    # 偏差的标准差
z_value = (resid - meanResid) / stdResid  # 计算偏差的z值

# 异常值
outliers_idx = z_value[abs(z_value) > 2].index  # z值大于n（设为2）判定为异常值

# 绘图
plt.scatter(y, y_pred, color='C0')
plt.scatter(y[outliers_idx], y_pred[outliers_idx], color='C1')
plt.grid()
plt.show()