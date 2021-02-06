import sys
sys.path.append('adpst')
import logging
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow.compat.v1 as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from style_transfer import *
from Func_face_detection import *
from Func_seg import *


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", type=str, help="content image path", default="content.png")
    parser.add_argument("--style_image", type=str, help="style image path", default="style.png")
    parser.add_argument("--output_image", type=str, help="Output image path, default: result.jpg",
                        default="result.jpg")
    parser.add_argument("--resize", type=int, help="Resize size, default: 256",
                        default=256)
    parser.add_argument("--iterations", type=int, help="Number of iterations, default: 4000",
                        default=500)
    parser.add_argument("--intermediate_result_interval", type=int,
                        help="Interval of iterations until a intermediate result is saved., default: 100",
                        default=100)
    parser.add_argument("--print_loss_interval", type=int,
                        help="Interval of iterations until the current loss is printed to console., default: 1",
                        default=10)
    parser.add_argument("--content_weight", type=float,
                        help="Weight of the content loss., default: 1",
                        default=1e-2)
    parser.add_argument("--style_weight", type=float,
                        help="Weight of the style loss., default: 100",
                        default=180)
    parser.add_argument("--regularization_weight", type=float,
                        help="Weight of the photorealism regularization.",
                        default=10 ** 2)
    parser.add_argument("--nima_weight", type=float,
                        help="Weight for nima loss.",
                        default=10 ** 2)
    parser.add_argument("--adam_learning_rate", type=float,
                        help="Learning rate for the adam optimizer., default: 1.0",
                        default=10.0)
    parser.add_argument("--adam_beta1", type=float,
                        help="Beta1 for the adam optimizer., default: 0.9",
                        default=0.9)
    parser.add_argument("--adam_beta2", type=float,
                        help="Beta2 for the adam optimizer., default: 0.999",
                        default=0.99)
    parser.add_argument("--adam_epsilon", type=float,
                        help="Epsilon for the adam optimizer., default: 1e-08",
                        default=1e-08)
    parser.add_argument("--semantic_thresh", type=float, help="Semantic threshold for label grouping., default: 0.8",
                        default=0.8)
    parser.add_argument("--similarity_metric", type=str, help="Semantic similarity metric for label grouping., default: li",
                        default="li")
    init_image_options = ["noise", "content", "style"]
    similarity_metric_options = ["li", "wpath", "jcn", "lin", "wup", "res"]
    parser.add_argument("--init", type=str, help="Initialization image (%s).", default="content")
    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.", default="0")

    args = parser.parse_args()
    assert (args.init in init_image_options)
    # For more information on the similarity metrics: http://gsi-upm.github.io/sematch/similarity/#word-similarity
    assert (args.similarity_metric in similarity_metric_options)
    res = args.resize
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    final_dir = join('res', 'final_result')
    temp = join('res', 'temp')
    if not os.path.exists('res'):
        os.mkdir('res')
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    if not os.path.exists(temp):
        os.mkdir(temp)
    if not os.path.exists(join(temp,'crops')):
        os.mkdir(join(temp,'crops'))
    print('enterrrrrr cropping')
    cropped_image, bb_shapes, shape = face_detection(args.content_image, 'content', _res=res, tmp_path=join(temp, 'crops'))
    cropped_style, _, _ = face_detection(args.style_image, 'style', _res=res, tmp_path=join(temp, 'crops'))
    print('enterrrrrr segmentation')
    face_parsing(respth=temp, dspth=join(temp, 'crops'), cp='79999_iter.pth', resize_size=res)
    print(f"shape = {shape}")

    content_segmentation_filename = join(temp, 'content_crop.png')
    style_segmentation_filename = join(temp, 'style_crop.png')


    write_metadata(temp, args)

    """Check if image files exist"""
    for path in [args.content_image, args.style_image]:
        if path is None or not os.path.isfile(path):
            print("Image file {} does not exist.".format(path))
            exit(0)
    original = load_image(args.content_image)

    content_image = load_image(cropped_image)

    style_image = load_image(cropped_style)

    print("Load segmentation from files.")
    content_segmentation_image = cv2.cvtColor(cv2.resize(cv2.imread(content_segmentation_filename),
                                                         (res,res), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    style_segmentation_image = cv2.cvtColor(cv2.resize(cv2.imread(style_segmentation_filename),
                                          (res,res), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    content_segmentation_masks, masks_to_keep = extract_segmentation_masks(content_segmentation_image, flag=True)
    style_segmentation_masks = extract_segmentation_masks(style_segmentation_image)


    cv2.imwrite(change_filename(temp, args.content_image, '_seg', '.png'),
                reduce_dict(content_segmentation_masks, content_image))
    cv2.imwrite(change_filename(temp, args.style_image, '_seg', '.png'),
                reduce_dict(style_segmentation_masks, style_image))

    if args.init == "noise":
        random_noise_scaling_factor = 0.0001
        random_noise = np.random.randn(*content_image.shape).astype(np.float32)
        init_image = vgg.postprocess(random_noise * random_noise_scaling_factor).astype(np.float32)
    elif args.init == "content":
        init_image = content_image
    elif args.init == "style":
        init_image = style_image
    else:
        print("Init image parameter {} unknown.".format(args.init))
        exit(0)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    result = style_transfer(content_image, style_image, mask_for_tf(content_segmentation_masks),
                            mask_for_tf(style_segmentation_masks), init_image, temp, args)[0]



    result = np.clip(result, 0, 255.0)
    result = np.array(result, dtype='uint8')

    final = merge_images(content_image[0], result, masks_to_keep)
    transferred_image = cv2.resize(final, (shape[1], shape[0]))

    original = np.uint8(original)

    original[0, bb_shapes[0]: bb_shapes[1], bb_shapes[2]: bb_shapes[3], :] = transferred_image

    save_image(original, os.path.join(final_dir, f"{os.path.basename(args.content_image)[:-4]}_to_{os.path.basename(args.style_image)[:-4]}.png"))

