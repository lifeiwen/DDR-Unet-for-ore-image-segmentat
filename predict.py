from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import save_img
import os
import numpy as np

  
def weighted_adaptive_loss(pred, target):
    """
    权重自适应损失函数
    
    参数：
    - pred：预测的分割结果，形状为 (batch_size, num_classes, height, width)
    - target：真实的分割标签，形状为 (batch_size, height, width)，取值为0或1
    
    返回：
    - 损失值
    """
    
    # 计算前景和背景的像素数量
    num_foreground_pixels = torch.sum(target == 1).float()
    num_background_pixels = torch.sum(target == 0).float()
    
    # 计算前景和背景的像素比例
    foreground_ratio = num_foreground_pixels / (num_foreground_pixels + num_background_pixels)
    background_ratio = num_background_pixels / (num_foreground_pixels + num_background_pixels)
    
    # 根据像素比例计算前景和背景的权重
    foreground_weight = 1.0 / (background_ratio + 1e-8)  # 添加一个小的值以避免除以零
    background_weight = 1.0 / (foreground_ratio + 1e-8)
    
    # 将权重应用于损失计算
    loss = torch.zeros_like(pred[:, 0, :, :])  # 初始化损失矩阵
    
    # 计算前景的损失
    foreground_loss = F.binary_cross_entropy_with_logits(pred[:, 1, :, :], target)
    loss[target == 1] = foreground_weight * foreground_loss
    
    # 计算背景的损失
    background_loss = F.binary_cross_entropy_with_logits(pred[:, 0, :, :], target)
    loss[target == 0] = background_weight * background_loss
    
    # 返回平均损失
    weighted_adaptive_loss = torch.mean(loss)
    return weighted_adaptive_loss

model = load_model('./model/unet_bestfocal_model.h5', custom_objects={'weighted_adaptive_loss': focal_loss()})
# 加载模型
#model = load_model('./model/unet_bestfocal_model.h5')

# 设置文件夹路径
input_folder = './dataset-fem/test/'
output_folder = './model/predictions-unet_bestfocal_model-fem/'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 加载图像并预处理
        img = image.load_img(os.path.join(input_folder, filename), target_size=(1024, 1280))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # 预测图像
        prediction = model.predict(x)

        # 处理预测结果
        prediction = prediction[0]
        prediction = (prediction > 0.5).astype(np.uint8)

        # 生成输出文件路径
        output_filename = os.path.join(output_folder, filename.replace(".png", "-pre.png"))

        # 保存预测结果
        save_img(output_filename, prediction)

print("预测完成并保存到文件夹：", output_folder)
