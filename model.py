from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

def residual_block(x, filters):
    """
    定义残差块
    """
    conv1 = Conv2D(filters, 3, activation='relu', padding='same')(x)
        
    conv2 = Conv2D(filters, 3, activation=None, padding='same')(conv1)
    conv1x1 = Conv2D(filters, 1, activation=None, padding='same')(x)  # 添加1x1的卷积操作
    add = Add()([conv1x1, conv2])
    out = Activation('relu')(add)
    return out

def dense_block(x, filters, n_layers):
    """
    定义密集块
    """
    concat = x
    for _ in range(n_layers):
        out = residual_block(concat, filters)
        concat = Concatenate()([concat, out])
    return concat

def downsample_block(x, filters):
    """
    定义下采样块
    """
    conv = Conv2D(filters, 3, strides=2, activation='relu', padding='same')(x)
    out = residual_block(conv, filters)
    return out

def upsample_block(x, skip_connection, filters):
    """
    定义上采样块
    """
    upsample = Conv2DTranspose(filters, 3, strides=2, activation='relu', padding='same')(x)
    concat = Concatenate()([upsample, skip_connection])
    out = residual_block(concat, filters)
    return out

def ResidualDenseUNet(input_shape):
    """
    定义Residual Dense U-Net模型
    """
    inputs = Input(input_shape)

    # 下采样
    down1 = downsample_block(inputs, 64)
    down2 = downsample_block(down1, 128)
    down3 = downsample_block(down2, 256)

    # 中间密集块
    dense_block1 = dense_block(down3, 256, 4)
    dense_block2 = dense_block(dense_block1, 256, 4)

    # 上采样
    up1 = upsample_block(dense_block2, down2, 128)
    up2 = upsample_block(up1, down1, 64)
    up3 = upsample_block(up2, inputs, 32)

    # 输出层
    outputs = Conv2D(1, 1, activation='sigmoid')(up3)

    model = Model(inputs=inputs, outputs=outputs)
    return model
