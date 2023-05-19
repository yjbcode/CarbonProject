import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# 数据
data = pd.Series(np.random.randn(10000)*100)

"1. 四分位数判别法" """
 a) 分位差(interquartile range, IQR) = 3/4位数 - 1/4位数 
 b) 判别标准：
    · 下限 = 1/4位数 - 1.5 * IQR
    · 上限 = 3/4位数 + 1.5 * IQR
    · 异常值：低于下限或高于上限
"""
# 统计量
q25 = data.quantile(q=0.25)  # 1/4位数
q75 = data.quantile(q=0.75)  # 3/4位数
iqr = q75 - q25  # 分位差
l_lim = q25 - 1.5 * iqr  # 下限
h_lim = q75 + 1.5 * iqr  # 上限

# 判别
abnormal = data[(data < l_lim) | (data > h_lim)]  # 异常值
normal = data[(data >= l_lim) & (data <= h_lim)]  # 正常值
print(f'异常值个数：{len(abnormal)} ')

# 箱型图
fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(2,1,1)
data.plot.box(vert=False, grid=True, ax = ax1)
plt.show()

# 图示结果
ax2 = fig.add_subplot(2,1,2)
plt.scatter(normal.index,normal,color = 'k',marker='.',alpha = 0.3)
plt.scatter(abnormal.index,abnormal,color = 'r',marker='.',alpha = 0.5)
plt.xlim([-10,10010])
plt.grid()
plt.show()



"2. 五九五分位数判别法"
# 统计量
l_lim = data.quantile(q=0.05)
h_lim = data.quantile(q=0.95)

# 判别
abnormal = data[(data < l_lim) | (data > h_lim)]  # 异常值
normal = data[(data >= l_lim) & (data <= h_lim)]  # 正常值
print(f'异常值个数：{len(abnormal)} ')

# 图示结果
ax2 = fig.add_subplot(1,1,1)
plt.scatter(normal.index,normal,color = 'k',marker='.',alpha = 0.3)
plt.scatter(abnormal.index,abnormal,color = 'r',marker='.',alpha = 0.5)
plt.xlim([-10,10010])
plt.grid()
plt.show()