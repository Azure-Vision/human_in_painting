from harmonize import *
from make_naive import *
import streamlit as st


def get_human_img_path():
    """We recommend that you upload a photo of person wearing colorful clothes, which will give better results in general."""
    own = st.checkbox(
        "Upload my own image (if not, we will provide default images.)")
    if own:
        pass
    default_paintings = get_default_paintings()
    image = st.selectbox("Choose human image:", list(default_paintings.keys()))
    # return "Images/default_human/man1.jpg"
    st.image(default_paintings[image])
    return default_paintings[image]


def get_painting_img_path():
    return "Images/default_paintings/scream.jpg"


def get_min_size():
    """Our pretrained models use min_size 120. bigger size means bigger resolution, but longer training time"""
    number = st.number_input(
        "Min size: (Our pretrained models use min_size 120. Bigger size implies bigger resolution, but longer training time.)", 120)
    return number  # 120


def get_gpu():
    """Get the GPU device."""
    number = st.number_input("GPU device:", 0)
    return number


def get_not_cuda():
    # res = st.checkbox("Do not use GPU", True)
    return True


def get_finetune():
    """Whether to finetune or not. Finetuning requires longer processing time (requires training even for default images), might output more blurry image. We suggest that use finetune for little_girl.png, do not use finetune for woman.png."""
    res = st.checkbox("Finetune after training. (Finetuning requires longer processing time (requires training even for default images). Resulting image might be better, but also might be worse, eg.  more blurry.)")
    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive_img_path",
                        default=None)
    args = parser.parse_args()
    args.src_img_path = get_painting_img_path()
    args.human_img_path = get_human_img_path()
    args.input_name = args.src_img_path
    args.train_mode = "harmonization"
    return args


def get_more_args(args):
    args.min_size = get_min_size()
    args.gpu = get_gpu()
    args.not_cuda = get_not_cuda()
    args.no_finetune = not get_finetune()
    return args


def get_default_paintings():
    images = {}
    rootdir = "Images/default_paintings"
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:  # 输出文件信息
            # 该文件路径为 os.path.join(rootdir, filename)
            images[filename] = os.path.join(rootdir, filename)
    return images


def get_default_human():
    images = {}
    rootdir = "Images/default_human"
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:  # 输出文件信息
            # 该文件路径为 os.path.join(rootdir, filename)
            images[filename] = os.path.join(rootdir, filename)
    return images


if __name__ == "__main__":
    st.write('Hello, world!')
    args = get_args()
    if not args.naive_img_path:
        human_segmentation, mask_image = segment_human(args.human_img_path)
        args.naive_img_path = get_naive_and_mask(
            human_segmentation, mask_image, args.src_img_path)
    print(f"naive image is saved in {args.naive_img_path}")
    args = get_more_args(args)
    train_model(args)
    model_dir, src_img_name = harmonize(args)
    result_img_dir = get_img(args, model_dir, src_img_name)
    print(f"Output image is saved in {result_img_dir}")
