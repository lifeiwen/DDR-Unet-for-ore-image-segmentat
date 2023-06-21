import os
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 设置数据集路径
data_path = "./dataset/"

# 加载图像和掩码
def load_data(path):
    images_path = os.path.join(path, "imgs")
    masks_path = os.path.join(path, "masks")

    images = []
    masks = []
    image_filenames = []

    # 遍历文件夹中的图像文件
    for filename in os.listdir(images_path):
        if filename.endswith(".png"):
            # 加载图像
            image = tf.keras.preprocessing.image.load_img(os.path.join(images_path, filename), color_mode='rgb')
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            image_filenames.append(filename)
            # 加载对应的掩码
            #mask_filename = filename.split("_")[0]
            #mask_filename = "_".join(filename.split("_")[:2])
            mask_filename = filename.replace("RLM", "Ref")
            mask = tf.keras.preprocessing.image.load_img(os.path.join(masks_path, mask_filename), color_mode='grayscale')
            mask = tf.keras.preprocessing.image.img_to_array(mask)
            masks.append(mask)

    return np.array(images), np.array(masks), image_filenames

# 加载数据集
x, y, filenames = load_data(data_path)

# 将像素值缩放到 0-1 范围内
x = x / 255.0
y = y / 255.0

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
