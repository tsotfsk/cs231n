import numpy as np 

a = np.full((2,2), 9)  # 用9填满一个2*2的矩阵 
                       # [9 9
                       #  9 9]

a = np.array([[1, 2], [3, 4]])  # 初始化时的列表嵌套表示矩阵
                                # [1 2
                                #  3 4]

print(a[np.arange(2), [0, 1]])  # 第一个参数是行的列表，第二个参数是列的列表
                             # [1 4]

print(a.T)  # a的转置
            # [1 3 
            #  2 4]
print(np.empty_like(a))  # 创建一个和a.shape相同的矩阵，值可能随机
                         # [9 9
                         #  9 9]

print(np.tile(a, (4, 2)))  # 行重复四次，列重复两次，just like
                           #  [a a
                           #   a a 
                           #   a a
                           #   a a]

print(np.reshape(a, (1, 4)))  # 将2*2的矩阵变为1*4的
                              # [1 2 3 4]

print(np.dot(a, a))  # 点乘
                     # [ 7 10
                     #  15 22]

print(a * a)  # 数乘
              # [1 4
              #  9 16]

a = np.random.random((2,2))  # 随机一个2*2矩阵，范围是(0,1)

if __name__ == "__main__":
    # TODO problem 多维矩阵的乘法，为什么2*2 dot 2*2的结果是二维，2*2*2 dot 2*2*2的结果是四维
    a = np.full((2, 2, 2, 2, 2), 2)
    b = np.full((2, 2, 2, 2, 2), 3)
    print(a.dot(b).shape)
    print(a.dot(b))