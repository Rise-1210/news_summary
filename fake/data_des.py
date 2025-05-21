import pandas as pd
import os

# 读取pickle文件
file_path = r'C:\Users\zhr\Desktop\summary\fake\data\test.pkl'
data = pd.read_pickle(file_path)

# 打印数据基本信息
print("数据形状:", data.shape)
print("\n数据前5行:")
print(data.head())

# 打印数据类型信息
print("\n数据类型信息:")
print(data.dtypes)

# 打印基本统计信息
print("\n基本统计信息:")
print(data.describe())
