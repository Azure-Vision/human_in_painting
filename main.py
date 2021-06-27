from harmonize import *
from make_naive import *
import streamlit as st
from PIL import Image
import time
import random


def get_uploaded_file(img_file_buffer, name):
    """把streamlit上传的文件保存在本地并返回地址"""
    # name = img_file_buffer.name
    name = "Images/tmp/" + name
    img_file = open(name, "wb")
    img_file.write(img_file_buffer.getvalue())
    return name


def get_human_img_path():
    """We recommend that you upload a photo of person wearing colorful clothes, which will give better results in general."""
    own = st.sidebar.checkbox(
        "Upload my own human image (if not, we will provide default images.)")
    if own:
        img_file_buffer = st.sidebar.file_uploader(
            "Upload an image (under 1 MB) (ignore the bugs if appear) ", type=["jpg", "jpeg"])
        time.sleep(15)
        path = get_uploaded_file(img_file_buffer, "someone.jpg")
        st.image(path, "human image", 300)
        return path
    else:
        default_paintings = get_default_human()
        image = st.sidebar.selectbox(
            "Choose human image:", list(default_paintings.keys()), index=2)
        st.image(default_paintings[image], "human image", 300)
        return default_paintings[image]


def get_painting_img_path():
    own = st.sidebar.checkbox(
        "Upload my own painting image (if not, we will provide default images.)")
    if own:
        img_file_buffer = st.sidebar.file_uploader(
            "Upload an image (under 1 MB) (ignore the bugs if appear) (note that for this image, we need to train from scratch, which requires GPU)", type=["jpg", "jpeg"])
        time.sleep(15)
        path = get_uploaded_file(img_file_buffer, "some_painting.jpg")
        st.image(path, "painting image", 300)
        return path
    else:
        default_paintings = get_default_paintings()
        image = st.sidebar.selectbox(
            "Choose painting image:", list(default_paintings.keys()), index=7)
        # return "Images/default_human/man1.jpg"
        st.image(default_paintings[image], "painting image", 300)
        # print(default_paintings[image])
        return default_paintings[image]


def get_min_size():
    """Our pretrained models use min_size 120. bigger size means bigger resolution, but longer training time"""
    number = st.sidebar.number_input(
        "Min size: (Pretrained models use 120. Bigger size implies better resolution, but longer training time.)", 120)
    return number  # 120

#


def get_gpu():
    """Get the GPU device."""
    return 0


def get_not_cuda():
    res = st.sidebar.checkbox("Do not use GPU", True)
    return res


def get_finetune():
    """Whether to finetune or not. Finetuning requires longer processing time (requires training even for default images), might output more blurry image. We suggest that use finetune for little_girl.png, do not use finetune for woman.png."""
    res = st.sidebar.checkbox(
        "Finetune after training. (Finetuning requires longer processing time. Resulting image might be better, but also might be worse, eg. more blurry.)")
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
    st.sidebar.markdown(
        "Custom setting: make sure you have cuda devices if you want to modify it!")
    args.min_size = get_min_size()
    args.gpu = 0
    args.not_cuda = False
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
    args = get_args()
    if not args.naive_img_path:
        human_segmentation, mask_image = segment_human(args.human_img_path)
        # st.image(human_segmentation, "human segmentation", 300)
        args.naive_img_path = get_naive_and_mask(
            human_segmentation, mask_image, args.src_img_path)
    print(f"naive image is saved in {args.naive_img_path}")
    args = get_more_args(args)
    train_model(args)
    model_dir, src_img_name = harmonize(args)
    result_img_dir = get_img(args, model_dir, src_img_name)
    print(f"Output image is saved in {result_img_dir}")
    st.write("He/She is placed into the painting!")
    st.image(result_img_dir, "result image", 300)
