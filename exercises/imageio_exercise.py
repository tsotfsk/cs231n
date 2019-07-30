import numpy as np
from PIL import Image
from imageio import imread, imwrite  # scipy新版本不再支持imread,imsave,imresize
import matplotlib.pyplot as plt

# 读取图像
img = imread('./exercises/test.png')
print(img.dtype, img.shape)

# 调整图像
img_tinted = img * [1, 0.95, 0.1]  # 数乘
img_tinted = np.array(Image.fromarray(np.uint8(img_tinted)).resize((300, 300)))  # np.uint8用作对矩阵中元素取整，否则会报错

imwrite('./exercises/test_result.png', img_tinted)

# 显示原始图像
plt.subplot(1, 2, 1)  # 一行两列的第一列
plt.imshow(img)

# 显示更改后图像
plt.subplot(1, 2, 2)  # 一行两列的第二列
plt.imshow(img_tinted)

plt.show()