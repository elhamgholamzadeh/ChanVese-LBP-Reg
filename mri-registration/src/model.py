import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Input, UpSampling3D, concatenate, LeakyReLU
from tensorflow.keras.initializers import RandomNormal

from .spatial_transformer import Dense3DSpatialTransformer


def my_conv(x_in, nf, strides=1):
    x_out = Conv3D(
        nf,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        strides=strides,
    )(x_in)
    return LeakyReLU(0.2)(x_out)


def unet_statistic(vol_size, enc_nf, dec_nf, full_size=True):
    src = Input(shape=vol_size + (1,), name="moving_image")
    tgt = Input(shape=vol_size + (1,), name="fixed_image")
    src_label = Input(shape=vol_size + (1,), name="label_data")
    tgt_label = Input(shape=vol_size + (1,), name="label_temp")

    x_in = concatenate([src, tgt])

    x0 = my_conv(x_in, enc_nf[0], 2)
    x1 = my_conv(x0, enc_nf[1], 2)
    x2 = my_conv(x1, enc_nf[2], 2)
    x3 = my_conv(x2, enc_nf[3], 2)

    x = my_conv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x2])

    x = my_conv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x1])

    x = my_conv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x0])

    x = my_conv(x, dec_nf[3])
    x = my_conv(x, dec_nf[4])

    if full_size:
        x = UpSampling3D()(x)
        x = concatenate([x, x_in])
        x = my_conv(x, dec_nf[5])

        if len(dec_nf) == 8:
            x = my_conv(x, dec_nf[6])

    flow = Conv3D(
        dec_nf[-1],
        kernel_size=3,
        padding="same",
        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
        name="flow",
    )(x)

    warped_src = Dense3DSpatialTransformer(name="warped_image")([src, flow])
    warped_label = Dense3DSpatialTransformer(name="warped_label")([src_label, flow])

    output_unet = concatenate([warped_src, warped_label], axis=-1, name="warped_image_and_label")

    return Model(
        inputs=[src, tgt, src_label, tgt_label],
        outputs=[output_unet, flow],
        name="unet_statistic",
    )
