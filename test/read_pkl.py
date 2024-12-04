import pickle

# 读取 .pkl 文件
with open('/data/groups/QY_LLM_Other/meisen/dataset/dpr/nq/downloads/data/reader/nq/single/test.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印读取的数据
print(data)
