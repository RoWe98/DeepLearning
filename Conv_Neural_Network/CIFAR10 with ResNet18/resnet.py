
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets


# 残差模块ResBlock
from tensorflow.python.keras import Sequential

'''
                        F(x)
                         ReLU                      H(x)=F(x)+x
x-------->Conv2d(64,3x3)------>Conv2d(64,3x3)------->(+)------->
      |                                               ^
      |                                               |
      |_______________________________________________|
                      identity(x)
                    残差模块  BasicBlock
'''

class BasicBlock(layers.Layer):

    # 残差模块类
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # f(x)包含了两个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn2 = layers.BatchNormalization()
        # 当F(x)和x形状不同时，无法相加，需要新建identity(x)卷积层，来完成x的形状转换
        if stride != 1: # 插入identity层
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
        else :
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        # 向前计算函数
        # [b, h, w, c], 通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        # 通过identity模块
        identity = self.downsample(inputs)
        # 2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output) # 激活函数
        return output


class ResNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes=10): # [2, ,2 ,2 ,2]
        super(ResNet, self).__init__()
        # 根网络,预处理
        self.stem = Sequential([
            layers.Conv2D(64, (3,3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1),padding='same'),
        ])
        # 堆叠 4 个 Block，每个 block 包含了多个 BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 通过 Pooling 层将高宽降低为 1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 前向计算函数：通过根网络
        x = self.stem(inputs)
        # 一次通过 4 个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):  # 其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def resnet18():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])


def resnet34():
     # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3])