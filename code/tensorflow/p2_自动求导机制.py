import tensorflow as tf

if __name__ == '__main__':
    x = tf.Variable(initial_value=3.)  # x 是一个初始化为 3 的 变量
    with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
        y = tf.square(x)  # tf.square() 操作代表对输入张量的每一个元素求平方，不改变张量形状
    y_grad = tape.gradient(y, x)  # 计算y关于x的导数
    print(y, y_grad)

    X = tf.constant([[1., 2.], [3., 4.]])
    y = tf.constant([[1.], [2.]])
    w = tf.Variable(initial_value=[[1.], [2.]])
    b = tf.Variable(initial_value=1.) 
    with tf.GradientTape() as tape:
        L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))  # tf.reduce_sum() 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量
    w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
    print(L, w_grad, b_grad)
