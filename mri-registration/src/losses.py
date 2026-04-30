import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

K = keras.backend
mse = tf.keras.losses.MeanSquaredError()


def gradient_loss(penalty="l1"):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if penalty == "l2":
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        return (tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz)) / 3.0

    return loss


def tf_lbp(y):
    y00 = y[:, 0:-2, 0:-2, :]
    y01 = y[:, 0:-2, 1:-1]
    y02 = y[:, 0:-2, 2:]
    y10 = y[:, 1:-1, 0:-2]
    y11 = y[:, 1:-1, 1:-1]
    y12 = y[:, 1:-1, 2:]
    y20 = y[:, 2:, 0:-2]
    y21 = y[:, 2:, 1:-1]
    y22 = y[:, 2:, 2:]

    z = tf.cast(tf.greater_equal(y01, y11), tf.uint8) * tf.constant(1, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y02, y11), tf.uint8) * tf.constant(2, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y12, y11), tf.uint8) * tf.constant(4, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y22, y11), tf.uint8) * tf.constant(8, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y21, y11), tf.uint8) * tf.constant(16, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y20, y11), tf.uint8) * tf.constant(32, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y10, y11), tf.uint8) * tf.constant(64, dtype=tf.uint8)
    z += tf.cast(tf.greater_equal(y00, y11), tf.uint8) * tf.constant(128, dtype=tf.uint8)
    return tf.cast(z, dtype=tf.uint8)


def tf_hist(tf_image):
    return tf.histogram_fixed_width(
        tf_image,
        [0.0, 1.0],
        nbins=1000,
        dtype=tf.dtypes.int32,
    )


def new_lbp_loss(y_true, y_pred):
    lbp_true = tf_lbp(y_true)
    lbp_pred = tf_lbp(y_pred)

    max_true = tf.cast(tf.math.reduce_max(y_true), dtype=tf.float32)
    max_pred = tf.cast(tf.math.reduce_max(y_pred), dtype=tf.float32)

    lbp_true = tf.cast(lbp_true, dtype=tf.float32) / (max_true + 1e-8)
    lbp_pred = tf.cast(lbp_pred, dtype=tf.float32) / (max_pred + 1e-8)

    return tf.cast(mse(tf_hist(lbp_true), tf_hist(lbp_pred)), dtype=tf.float32)


def correlation_lbp(y_true, y_pred):
    return tfp.stats.correlation(
        y_true,
        y_pred,
        sample_axis=None,
        event_axis=None,
        keepdims=False,
    )


def get_uniques(t):
    t1d = tf.reshape(t, shape=(-1,))
    uniques, _, _ = tf.unique_with_counts(t1d)
    return uniques


def statistical(temp, move, label_temp, label_move, vol_size=(160, 192, 224)):
    def condition(i, temp, move, label_temp, label_move, label, mean_segm_shape):
        return i < tf.shape(label)[0]

    def body(i, temp, move, label_temp, label_move, label, mean_segm_shape):
        value = label[i]

        def add_loss():
            label_find = K.equal(label_temp, value)
            label_find1 = K.equal(label_move, value)
            label_xor = tf.math.logical_xor(label_find, label_find1)

            cast_label_find = tf.cast(label_find, dtype=tf.float32)
            local_temp = tf.math.multiply(temp, cast_label_find)
            count = tf.math.count_nonzero(label_find, dtype=tf.float32)

            mean_shape_slice = tf.reduce_sum(local_temp) / (count + 1e-8)

            cast_xor = tf.cast(label_xor, dtype=tf.float32)
            local_move = tf.math.multiply(move, cast_xor)
            eq = tf.square(local_move - mean_shape_slice)

            return mean_segm_shape + tf.cast(tf.reduce_sum(eq), dtype=tf.float32)

        mean_segm_shape = tf.cond(
            tf.not_equal(value, tf.cast(1, value.dtype)),
            add_loss,
            lambda: mean_segm_shape,
        )

        return [i + 1, temp, move, label_temp, label_move, label, mean_segm_shape]

    h = tf.constant(0, dtype=tf.float32)
    j = tf.constant(0, dtype=tf.int32)
    label = get_uniques(label_temp)

    _, _, _, _, _, _, statistical_loss = tf.while_loop(
        cond=condition,
        body=body,
        loop_vars=[j, temp, move, label_temp, label_move, label, h],
    )

    return statistical_loss / float(vol_size[0] * vol_size[1] * vol_size[2])


def m_loss(win=(9, 9, 9), voxel_weights=None, vol_size=(160, 192, 224)):
    def main_loss(i_input, j_input):
        y_fix_source = i_input[:, :, :, :, :1]
        label_target = i_input[:, :, :, :, -1:]

        y_fix_source_pred = j_input[:, :, :, :, :1]
        label_target_pred = j_input[:, :, :, :, -1:]

        i = i_input[:, :, :, :, :1]
        j = j_input[:, :, :, :, :1]

        i2 = i * i
        j2 = j * j
        ij = i * j

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        i_sum = tf.nn.conv3d(i, filt, [1, 1, 1, 1, 1], "SAME")
        j_sum = tf.nn.conv3d(j, filt, [1, 1, 1, 1, 1], "SAME")
        i2_sum = tf.nn.conv3d(i2, filt, [1, 1, 1, 1, 1], "SAME")
        j2_sum = tf.nn.conv3d(j2, filt, [1, 1, 1, 1, 1], "SAME")
        ij_sum = tf.nn.conv3d(ij, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0] * win[1] * win[2]
        u_i = i_sum / win_size
        u_j = j_sum / win_size

        cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
        i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * win_size
        j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * win_size

        cc = cross * cross / (i_var * j_var + 1e-5)

        loss_statistical = statistical(
            y_fix_source[..., 0],
            y_fix_source_pred[..., 0],
            label_target[..., 0],
            label_target_pred[..., 0],
            vol_size=vol_size,
        )

        return -1.0 * tf.reduce_mean(cc) + (1.0 / 19.0) * loss_statistical

    return main_loss
