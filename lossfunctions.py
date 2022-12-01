import tensorflow as tf
from read_images import content_layers, style_layers, num_content_layers, num_style_layers


def get_content_loss(base_content, target):
    # weight_per_content_layer = 1.0 / float(num_content_layers)
    # content_score = 0
    assert (len(base_content) == len(target))

    B, H, W, CH = base_content.get_shape()
    content_score = 2 * tf.nn.l2_loss(base_content - target) / (B * H * W * CH)
    return content_score


def get_style_loss(base_style, gram_target):
    # style_score=0
    # weight_per_style_layer = 1.0 / float(num_style_layers)
    # assert(len(base_style) == len(gram_target))

    B, H, W, CH = base_style.get_shape()
    G = _gram_matrix(base_style)
    A = gram_target
    # style_score +=  weight_per_style_layer * 2 * tf.nn.l2_loss(G - A) /  (B * (CH ** 2))
    style_score = 2 * tf.nn.l2_loss(G - A) / (B * (CH ** 2))
    return style_score


def _gram_matrix(input_tensor, shape=None):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = input_tensor.get_shape()
    num_locations = input_shape[1] * input_shape[2] * input_shape[3]
    num_locations = tf.cast(num_locations, tf.float32)
    return result / num_locations


def compute_loss(c_content, y_content, s_style, y_style, transformed_img, style_weight, content_weight,
                 variation_weight):
    content_score = 0
    style_score = 0
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(c_content, y_content):
        content_score += weight_per_content_layer * get_content_loss(comb_content, target_content)
    # Accumulate content losses from all layers

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(s_style, y_style):
        style_score += weight_per_style_layer * get_style_loss(comb_style, target_style)

    B, W, H, CH = transformed_img.get_shape()
    variation_score = tf.reduce_sum(tf.image.total_variation(transformed_img)) / (W * H)
    style_score *= style_weight
    content_score *= content_weight
    variation_score *= variation_weight
    # Get total loss
    loss = style_score + content_score + variation_score
    return loss
