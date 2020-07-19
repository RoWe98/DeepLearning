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


    # 通过build_resblock可以实现多过个残差模块的新建
    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数，堆叠 filter_num 个 BasicBlock
        res_blocks = Sequential()
        # 只有第一个 BasicBlock 的步长可能不为 1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):  # 其他 BasicBlock 步长都为 1
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks
