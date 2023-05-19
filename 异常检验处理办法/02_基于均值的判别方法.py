import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# 数据
data = pd.Series(np.random.randn(10000)*100)

"1. 3σ判别法" """
 a) 假设：数据服从正态分布
 b) 判别标准：异常值 < (μ-3σ) or 异常值 > (μ+3σ)
    其中，μ表示均值，σ表示标准差
"""
# 统计量
ave = data.mean()  # 均值
std = data.std()   # 标准差

# 判别
abnormal = data[np.abs(data - ave) >= 3*std]  # 异常值
normal = data[np.abs(data - ave) <= 3*std]    # 正常值
print(f'异常值个数: {len(abnormal)} 个')

# 图示结果
plt.scatter(normal.index, normal, color='k', marker='.', alpha = 0.3)
plt.scatter(abnormal.index, abnormal, color='r', marker='.', alpha = 0.5)
plt.xlim([-10,10010])
plt.grid()
plt.show()


"2. z值判别法" """
 a) 假设：数据服从正态分布
 b) 判别标准：将数据x转换为z值，(x-μ)/σ
    z值反映了数据x到均值μ相差多少倍标准差。如果倍数过大，判定为异常值
 c) 可以看出，z值判别法和3σ判别法本质上相同
"""
# z值
z_value = (data - np.mean(data))/np.std(data)

# 判别
abnormal = data[abs(z_value) >= 2]  # 异常值（假定超过2倍标准差为异常值）
normal = data[abs(z_value) < 2]     # 正常值
print(f'异常值个数: {len(abnormal)} 个')

# 图示结果
plt.scatter(normal.index, normal, color='k', marker='.', alpha = 0.3)
plt.scatter(abnormal.index, abnormal, color='r', marker='.', alpha = 0.5)
plt.xlim([-10,10010])
plt.grid()
plt.show()