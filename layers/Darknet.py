import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, LeakyReLU, BatchNormalization
#from layers.batch_nomalize import BatchNormalization
from tensorflow.keras.regularizers import l2
from itertools import repeat

def DarknetConv(x, filters, size, strides=1, batch_norm=True, activate_type = "leaky"):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'

    x = Conv2D(filters=filters, 
                kernel_size=size,
                strides=strides, 
                padding=padding,
                use_bias=not batch_norm, 
                kernel_regularizer=l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        #x = LeakyReLU(alpha=0.1)(x)
        if activate_type == "leaky":
            x = LeakyReLU(alpha=0.1)(x)
        elif activate_type == "mish":
            x = mish(x)

    
    return x

def DarknetResidual(x, filters, activate_type='leaky'):
    previous  = x
    x = DarknetConv(x, filters // 2, 1, activate_type=activate_type)
    x = DarknetConv(x, filters, 3, activate_type=activate_type)
    x = Add()([previous , x])
    return x

def DarknetBlock(x, filters, blocks, activate_type ='leaky'):
    x = DarknetConv(x, filters, 3, strides=2, activate_type = activate_type)
    for _ in repeat(None, blocks):
        x = DarknetResidual(x, filters, activate_type=activate_type)       
    return x

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)
    x = x_36 = DarknetBlock(x, 256, 8)
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def CSPDarknetResidual(x, filter_num1, filter_num2, activate_type='leaky'):
    previous  = x
    x = DarknetConv(x, filter_num1, 1, activate_type=activate_type)
    x = DarknetConv(x, filter_num2, 3, activate_type=activate_type)
    x = Add()([previous , x])
    return x

def CSPDarknetBlock(x, filter_num1, filter_num2, blocks, activate_type='leaky'):
    #x = DarknetConv(x, filters, 3, strides=2, activate_type)
    for _ in repeat(None, blocks):
        x = CSPDarknetResidual(x, filter_num1, filter_num2, activate_type=activate_type)       
    return x

def CSPDarknet53(name=None):
    x = inputs = Input([None, None, 3])

    x = DarknetConv(x, 32, 3 ,activate_type='mish')
    x = DarknetConv(x, 64, 3, strides = 2,activate_type='mish')
    route = x
    route = DarknetConv(route, 64, 1,activate_type='mish')
    x = DarknetConv(x, 64, 1,activate_type='mish')
    x = CSPDarknetBlock(x, 32, 64, 1)
    x = DarknetConv(x, 64, 1,activate_type='mish')
    x = tf.concat([x,route], axis=-1) 

    x = DarknetConv(x, 64, 1,activate_type='mish')
    x = DarknetConv(x, 128, 3,strides = 2,activate_type='mish')
    route = x
    route = DarknetConv(x, 64, 1,activate_type='mish')
    x = DarknetConv(x, 64, 1,activate_type='mish')
    x = CSPDarknetBlock(x, 64, 64, 2)
    x = DarknetConv(x, 64, 1,activate_type='mish')
    x = tf.concat([x,route], axis=-1) 

    x = DarknetConv(x, 128, 1,activate_type='mish')
    x = DarknetConv(x, 256, 3,strides = 2,activate_type='mish')
    route = x
    route = DarknetConv(x, 128, 1,activate_type='mish')
    x = DarknetConv(x, 128, 1,activate_type='mish')
    x = CSPDarknetBlock(x, 128, 128, 8)
    x = DarknetConv(x, 128, 1,activate_type='mish')
    x = tf.concat([x,route], axis=-1) 

    x = DarknetConv(x, 256, 1,activate_type='mish')
    route_1 = x
    x = DarknetConv(x, 512, 3, strides = 2,activate_type='mish')
    route = x
    route = DarknetConv(route, 256, 1,activate_type='mish')
    x = DarknetConv(x, 256, 1,activate_type='mish')
    x = CSPDarknetBlock(x, 256, 256, 8)
    x = DarknetConv(x, 256, 1,activate_type='mish')
    x = tf.concat([x,route], axis=-1)

    x = DarknetConv(x, 512, 1,activate_type='mish')
    route_2 = x
    x = DarknetConv(x, 1024, 3, strides = 2,activate_type='mish')
    route = x
    route = DarknetConv(route, 512, 1,activate_type='mish')
    x = DarknetConv(x, 512, 1,activate_type='mish')
    x = CSPDarknetBlock(x, 512, 512, 4)
    x = DarknetConv(x, 512, 1,activate_type='mish')
    x = tf.concat([x,route], axis=-1) 

    x = DarknetConv(x, 1024, 1,activate_type='mish')
    x = DarknetConv(x, 512, 1,activate_type='mish')
    x = DarknetConv(x, 1024, 3,activate_type='mish')
    x = DarknetConv(x, 512, 1,activate_type='mish')

    x = tf.concat([tf.nn.max_pool(x, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(x, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(x, ksize=5, padding='SAME', strides=1), x],axis=-1)

    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    return tf.keras.Model(inputs, (route_2, route_1, x), name=name)

def Upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)