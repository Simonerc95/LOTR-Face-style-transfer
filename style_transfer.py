import json
import os
import sys
sys.path.append('adpst')
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import components.VGG19.model as vgg
from components.matting import compute_matting_laplacian
from components.semantic_merge import *
import cv2
def style_transfer(content_image, style_image, content_masks, style_masks, init_image, result_dir, args):
    print("Style transfer started")

    content_image = vgg.preprocess(content_image)
    style_image = vgg.preprocess(style_image)

    weight_restorer = vgg.load_weights()

    image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    vgg19 = vgg.VGG19ConvSub(image_placeholder)

    with tf.Session() as sess:
        transfer_image = tf.Variable(init_image)
        transfer_image_vgg = vgg.preprocess(transfer_image)
        # transfer_image_nima = nima.preprocess(transfer_image)

        sess.run(tf.global_variables_initializer())
        weight_restorer.init(sess)
        content_conv4_2 = sess.run(fetches=vgg19.conv4_2, feed_dict={image_placeholder: content_image})
        style_conv1_1, style_conv2_1, style_conv3_1, style_conv4_1, style_conv5_1 = sess.run(
            fetches=[vgg19.conv1_1, vgg19.conv2_1, vgg19.conv3_1, vgg19.conv4_1, vgg19.conv5_1],
            feed_dict={image_placeholder: style_image})

        with tf.variable_scope("", reuse=True):
            vgg19 = vgg.VGG19ConvSub(transfer_image_vgg)

        content_loss = calculate_layer_content_loss(content_conv4_2, vgg19.conv4_2)

        style_loss = (1. / 5.) * calculate_layer_style_loss(style_conv1_1, vgg19.conv1_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv2_1, vgg19.conv2_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv3_1, vgg19.conv3_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv4_1, vgg19.conv4_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv5_1, vgg19.conv5_1, content_masks, style_masks)

        photorealism_regularization = calculate_photorealism_regularization(transfer_image_vgg, content_image)

        #nima_loss = compute_nima_loss(transfer_image_nima)

        content_loss = args.content_weight * content_loss
        style_loss = args.style_weight * style_loss
        photorealism_regularization = args.regularization_weight * photorealism_regularization
        #nima_loss = args.nima_weight * nima_loss

        total_loss = content_loss + style_loss + photorealism_regularization# + nima_loss

        tf.summary.scalar('Content loss', content_loss)
        tf.summary.scalar('Style loss', style_loss)
        tf.summary.scalar('Photorealism Regularization', photorealism_regularization)
        #tf.summary.scalar('NIMA loss', nima_loss)
        tf.summary.scalar('Total loss', total_loss)

        # TODO: Colab notebook (Demo) - Tuning
        # TODO: Paper:  automated-deep-photo-style-transfer

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(os.path.dirname(__file__), 'logs/'),
                                               sess.graph)

        iterations_dir = os.path.join(result_dir, "iterations")
        if not os.path.exists(iterations_dir):
            os.mkdir(iterations_dir)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.adam_learning_rate, beta1=args.adam_beta1,
                                           beta2=args.adam_beta2, epsilon=args.adam_epsilon)

        train_op = optimizer.minimize(total_loss, var_list=[transfer_image])
        sess.run(adam_variables_initializer(optimizer, [transfer_image]))

        min_loss, best_image = float("inf"), None
        for i in range(args.iterations + 1):
            _, result_image, loss, c_loss, s_loss, p_loss, summary = sess.run(
                fetches=[train_op, transfer_image, total_loss, content_loss, style_loss, photorealism_regularization,
                         summary_op])

            summary_writer.add_summary(summary, i)

            if i % args.print_loss_interval == 0:
                print(
                    "Iteration: {0:5} \t "
                    "Total loss: {1:15.2f} \t "
                    "Content loss: {2:15.2f} \t "
                    "Style loss: {3:15.2f} \t "
                    "Photorealism Regularization: {4:15.2f} \t "
                    .format(i, loss, c_loss, s_loss, p_loss))

            if loss < min_loss:
                min_loss, best_image = loss, result_image

            if i % args.intermediate_result_interval == 0:
                save_image(best_image, os.path.join(iterations_dir, "iter_{}.png".format(i)))

        return best_image


def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)
'''

def compute_nima_loss(image):
    model = nima.get_nima_model(image)

    def mean_score(scores):
        scores = tf.squeeze(scores)
        si = tf.range(1, 11, dtype=tf.float32)
        return tf.reduce_sum(tf.multiply(si, scores), name='nima_score')

    nima_score = mean_score(model.output)

    nima_loss = tf.identity(10.0 - nima_score, name='nima_loss')
    return nima_loss
'''

def calculate_layer_content_loss(content_layer, transfer_layer):
    return tf.reduce_mean(tf.squared_difference(content_layer, transfer_layer))


def calculate_layer_style_loss(style_layer, transfer_layer, content_masks, style_masks):
    # scale masks to current layer
    content_size = tf.TensorShape(transfer_layer.shape[1:3])
    style_size = tf.TensorShape(style_layer.shape[1:3])

    def resize_masks(masks, size):
        return [tf.image.resize_bilinear(mask, size) for mask in masks]

    style_masks = resize_masks(style_masks, style_size)
    content_masks = resize_masks(content_masks, content_size)

    feature_map_count = np.float32(transfer_layer.shape[3].value)
    feature_map_size = np.float32(transfer_layer.shape[1].value) * np.float32(transfer_layer.shape[2].value)

    means_per_channel = []
    for content_mask, style_mask in zip(content_masks, style_masks):
        transfer_gram_matrix = calculate_gram_matrix(transfer_layer, content_mask)
        style_gram_matrix = calculate_gram_matrix(style_layer, style_mask)

        mean = tf.reduce_mean(tf.squared_difference(style_gram_matrix, transfer_gram_matrix))
        means_per_channel.append(mean / (2 * tf.square(feature_map_count) * tf.square(feature_map_size)))

    style_loss = tf.reduce_sum(means_per_channel)

    return style_loss


def calculate_photorealism_regularization(output, content_image):
    # normalize content image and out for matting and regularization computation
    content_image = content_image / 255.0
    output = output / 255.0

    # compute matting laplacian
    matting = compute_matting_laplacian(content_image[0, ...])

    # compute photorealism regularization loss
    regularization_channels = []
    for output_channel in tf.unstack(output, axis=-1):
        channel_vector = tf.reshape(tf.transpose(output_channel), shape=[-1])
        matmul_right = tf.sparse_tensor_dense_matmul(matting, tf.expand_dims(channel_vector, -1))
        matmul_left = tf.matmul(tf.expand_dims(channel_vector, 0), matmul_right)
        regularization_channels.append(matmul_left)

    regularization = tf.reduce_sum(regularization_channels)
    return regularization


def calculate_gram_matrix(convolution_layer, mask):
    matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
    mask_reshaped = tf.reshape(mask, shape=[matrix.shape[0], 1])
    matrix_masked = matrix * mask_reshaped
    return tf.matmul(matrix_masked, matrix_masked, transpose_a=True)

def merge_images(img1_arr, img2_arr, masks):
    assert img1_arr.shape == img2_arr.shape, "shapes of the images have to be equal"
    assert type(masks) == dict
    H, W = img1_arr.shape[:-1]
    result = img2_arr.copy()
    Y = np.linspace(-1, 1, H)[None, :]
    X = np.linspace(-1, 1, W)[:, None]
    alpha = np.sqrt(X ** 6 + Y ** 2)
    alpha = 1 - np.clip(0, 1, alpha)
    channel_grads = np.stack([alpha,alpha,alpha], axis=2)
    result = result * channel_grads + img1_arr * (1-channel_grads)

    return result



def load_image(filename):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def save_image(image, filename):
    image = image[0, :, :, :]
    image = np.clip(image, 0, 255.0)
    image = np.uint8(image)

    result = Image.fromarray(image)
    result.save(filename)


def change_filename(dir_name, filename, suffix, extension=None):
    path, ext = os.path.splitext(filename)
    if extension is None:
        extension = ext
    return os.path.join(dir_name, path + suffix + extension)


def write_metadata(dir, args):
    # collect metadata and write to transfer dir
    meta = {
        "init": args.init,
        "iterations": args.iterations,
        "content": args.content_image,
        "style": args.style_image,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "regularization_weight": args.regularization_weight,
        "nima_weight": args.nima_weight,
        "semantic_thresh": args.semantic_thresh,
        "similarity_metric": args.similarity_metric,
        "adam": {
            "learning_rate": args.adam_learning_rate,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon
        }
    }
    filename = os.path.join(dir, "meta.json")
    with open(filename, "w+") as file:
        file.write(json.dumps(meta, indent=4))

