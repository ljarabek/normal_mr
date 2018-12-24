import tensorflow as tf


def AdaIN(content_features, style_features, alpha):
    '''
    Normalizes the `content_features` with scaling and offset from `style_features`.
    See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
    '''
    style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
    epsilon = 1e-5
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean,
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # MUNIT!!!
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta