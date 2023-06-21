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

# 定义Deformable U-Net模型
def deformable_unet(input_shape):
    inputs = Input(input_shape)

    conv1 = Convolution2DOffset(64, (3, 3), padding='same')(inputs)
    conv1 = Convolution2DOffset(64, (3, 3), padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2DOffset(128, (3, 3), padding='same')(pool1)
    conv2 = Convolution2DOffset(128, (3, 3), padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2DOffset(256, (3, 3), padding='same')(pool2)
    conv3 = Convolution2DOffset(256, (3, 3), padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2DOffset(512, (3, 3), padding='same')(pool3)
    conv4 = Convolution2DOffset(512, (3, 3), padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2DOffset(1024, (3, 3), padding='same')(pool4)
    conv5 = Convolution2DOffset(1024, (3, 3), padding='same')(conv5)

    up6 = Conv2D(512, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Convolution2DOffset(512, (3, 3), padding='same')(merge6)
    conv6 = Convolution2DOffset(512, (3, 3), padding='same')(conv6)

    up7 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Convolution2DOffset(256, (3, 3), padding='same')(merge7)
    conv7 = Convolution2DOffset(256, (3, 3), padding='same')(conv7)

    up8 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Convolution2DOffset(128, (3, 3), padding='same')(merge8)
    conv8 = Convolution2DOffset(128, (3, 3), padding='same')(conv8)

    up9 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Convolution2DOffset(64, (3, 3), padding='same')(merge9)
    conv9 = Convolution2DOffset(64, (3, 3), padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
