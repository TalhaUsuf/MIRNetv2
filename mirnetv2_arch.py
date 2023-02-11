import tensorflow as tf
from rich.console import Console



def ContextBlock(n_feat=80, bias=False):
    channel_add_conv = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(n_feat, 1, 1, input_shape=[1, 1, 80]), tf.keras.layers.LeakyReLU(alpha=0.2),
         tf.keras.layers.Conv2D(n_feat, 1, 1)])


    def modeling(x):
        # x -> [N, C, H, W]
        batch, channel, height, width = tf.shape(x) # -> [  2,  80, 256, 256]
        input_x = x

        input_x = tf.reshape(input_x, [batch, channel, height*width])# -> [    2,    80, 65536]

        input_x = tf.expand_dims(input_x, 1) # -> [2, 1, 80, 65536]

        context_mask = tf.keras.layers.Conv2D(1, 1)(tf.transpose(x, [0, 2, 3, 1]))# -> [2, 256, 256, 1]

        context_mask = tf.reshape(context_mask, [batch, height * width, 1])

        context_mask = tf.keras.layers.Softmax(axis=1)(context_mask) # -> [2, 65536, 1]

        context_mask = tf.expand_dims(context_mask, 1) # -> [2, 1, 65536, 1]

        context = tf.linalg.matmul(input_x, context_mask) # -> [2, 1, 80, 1]

        context = tf.reshape(context, [batch, channel, 1, 1]) # -> [2, 80, 1, 1]

        return context

    def forward(x): # channel last
        # x ->  [2, 256, 256, 80]
        x = tf.transpose(x, [0,3,1,2])
        # x ->  [  2,  80, 256, 256]
        context = modeling(x)
        context = tf.transpose(context, [0, 2,3,1])  # -> [2, 1, 1, 80]

        channel_add_term = channel_add_conv(context) # -> [2, 1, 1, 80]
        x = tf.transpose(x, [0, 2,3,1])  # -> [2, 256, 256, 80]
        x = x + channel_add_term # same -> [2, 256, 256, 80]
        # return -> [2, 256, 256, 80]
        return x # channel last

    return forward


def RCB():

    def body(feat, x):
        # model = tf.keras.Sequential([tf.keras.layers.Conv2D(feat, 3, 1, padding='same', input_shape=tf.shape(x)[1:]),
        #                             tf.keras.layers.LeakyReLU(0.2),
        #                             tf.keras.layers.Conv2D(feat, 3, 1, padding='same')])

        o = tf.keras.layers.Conv2D(feat, 3, 1, padding='same', input_shape=tf.shape(x)[1:])(x)
        o = tf.keras.layers.LeakyReLU(0.2)(o)
        o = tf.keras.layers.Conv2D(feat, 3, 1, padding='same')(o)
        return o

    def rcb(x, bias=False, groups=1):
        # x -> [N, H, W, C]

        feat = tf.shape(x)[-1]
        res = tf.keras.layers.Lambda(lambda feat, x: body(feat, x))(feat=feat, x=x)
        res = tf.keras.layers.LeakyReLU(0.2)(ContextBlock(n_feat=1)(res))
        res += x

        return res

    return rcb

x = tf.random.normal([4, 256, 256, 80])
# feat = tf.shape(x)[-1]
# context_block = ContextBlock(n_feat=1)
# out = context_block(x)
#
# Console().log(f"🔵 input shape --> {tf.shape(x).numpy()}")
# Console().log(f"🔴 output shape --> {tf.shape(out).numpy()}")
inputs = tf.keras.Input(shape=(256, 256, 80))
# out = ContextBlock(n_feat=1)(inputs)
out = RCB()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=out)

Console().print(model.summary())