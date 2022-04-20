import tensorflow as tf


class Linear(tf.keras.Model):
    # 初始化模型所需要的层
    def __init__(self):
        super().__init__()
        # 实例化一个全连接层
        self.dense = tf.keras.layers.Dense(
            units=1,  # 输出张量的维度
            activation=None,  # 如果不指定激活函数，即是纯粹的线性变换 AW + b
            kernel_initializer=tf.zeros_initializer(),  # 权重矩阵 kernel 初始化器
            bias_initializer=tf.zeros_initializer()  # 偏置向量 bias 初始化器
        )

    # 描述输入数据如何通过各种层得到输出
    def call(self, input):
        output = self.dense(input)
        return output


if __name__ == '__main__':
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])
    # 以下代码结构与前节类似
    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)  # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)
