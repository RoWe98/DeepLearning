import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses

log_dir = './tensorboard_img/'
summary_writer=tf.summary.create_file_writer(log_dir)

def preprocess(x, y):
    x=tf.cast(x, dtype=tf.float32)/ 255
    y=tf.cast(y, dtype=tf.int32)
    return x,y

batchsz=128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1), # 第一层卷积 6个3x3
    layers.MaxPooling2D(pool_size=2, strides=2), # 宽高减半的池化层
    layers.ReLU(), #激活函数
    layers.Conv2D(16, kernel_size=3, strides=1), # 第二层卷积 16个3x3
    layers.MaxPooling2D(pool_size=2, strides=2), # 宽高减半的池化层
    layers.ReLU(), #激活函数
    layers.Flatten(), # 打平层，方便全连接处理
    
    layers.Dense(120, activation='relu'), # 120个结点的全连接层
    layers.Dense(84, activation='relu'), # 全连接层 84节点
    layers.Dense(10), # 全连接层 10结点
])

network.build(input_shape=(4, 28, 28, 1))
# 统计网络信息
network.summary()

# 创建损失函数的类，在实际计算中调用类的实例即可
criteon=losses.CategoricalCrossentropy(from_logits=True)

optimizer = optimizers.Adam(lr=0.01)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for step, (x,y) in enumerate(db):
    
    # 构建梯度记录环境
    with tf.GradientTape() as tape:
        # 插入通道维度, =>[b,28,28,1]
        x=tf.expand_dims(x, axis=3) # 增加维度
        # 向前计算,获得10类别的概率分布 [b, 784] = > [b,10]
        out = network(x)
        # 真实标签One-hot编码 [b]=>[b,10]
        y_onehot = tf.one_hot(y, depth=10)
        # 计算交叉熵损失函数，标量
        loss = criteon(y_onehot, out)
        loss_meter.update_state(loss)
        
        with summary_writer.as_default(): # 写入环境
            tf.summary.scalar('train-loss', float(loss), step=step)
      
    # 自动计算梯度
    grads = tape.gradient(loss, network.trainable_variables)
    # 自动更新参数
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    
    if step%100==0:
        
        print(step, 'loss:', loss_meter.result().numpy())
        loss_meter.reset_states()
        
     # evaluate
    if step % 500 == 0:
        # 记录预测正确的数量， 总样本数量
        correct, total=0,0
        for step, (x,y) in enumerate(ds_val):
            # 插入通道维数, => [b, 28, 28, 1]
            x=tf.expand_dims(x, axis=3)
            # 向前计算, 获得10类别的预测分布, [b, 784]=>[b,10]
            out = network(x)
            # 真实流程是先经过softmax再argmax
            # 但是由于softmax不改变元素大小的相对关系，故省去
            pred=tf.argmax(out, axis=-1)
            y=tf.cast(y, tf.int64)
            # 统计预测正确数量
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
            # 统计预测样本总数
            total += x.shape[0]
            
            
        # 计算准确率
        print('test acc:',correct/total)
        
        val_images=x[:10]
        with summary_writer.as_default():
                tf.summary.scalar('test-acc', float(correct/total), step=step)
                tf.summary.image("val-onebyone-images:", val_images, max_outputs=9, step=step)