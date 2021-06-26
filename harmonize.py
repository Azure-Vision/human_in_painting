import os
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage import io
from Con_Sin_GAN.ConSinGAN.functions import generate_dir2save
import Con_Sin_GAN.main_train as train
import Con_Sin_GAN.evaluate_model as eval
import shutil


def write_html(args, title="Put Me Into The Paintings", content="", img=None):
    html_file = open("index.html", "w")
    img_line = ""
    if img:
        img_line = """<img src="{}" />""".format(
            os.path.join("Con_Sin_GAN", img))
    html_content = '''
    <head> 
        <h1> {} </h1> 
    </head>
    <body> 
        <h2> {} </h2>
        {}
        <h2> Source human image: </h2>
        <img src="{}" />
        <h2> Target painting image: </h2>
        <img src="{}" />
    </body>
    '''.format(title, content, img_line, os.path.join("Con_Sin_GAN", args.naive_img_path), os.path.join("Con_Sin_GAN", args.src_img_path))
    html_file.write(html_content)


def get_default_args():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.Dsteps = 3
    opt.Gsteps = 3
    opt.activation = 'lrelu'
    opt.alpha = 10
    opt.batch_norm = 0
    opt.beta1 = 0.5
    opt.fine_tune = False
    opt.gamma = 0.1
    opt.gpu = 0
    opt.input_name = None
    opt.ker_size = 3
    opt.lambda_grad = 0.1
    opt.lr_d = 0.0005
    opt.lr_g = 0.0005
    opt.lr_scale = 0.1
    opt.lrelu_alpha = 0.05
    opt.manualSeed = None
    opt.max_size = 250
    opt.min_size = 25
    opt.model_dir = None
    opt.naive_img = None
    opt.nc_im = 3
    opt.nfc = 64
    opt.niter = 2000
    opt.noise_amp = 0.1
    opt.not_cuda = True
    opt.num_layer = 3
    opt.padd_size = 0
    opt.start_scale = 0
    opt.train_depth = 3
    opt.train_mode = 'harmonization'
    opt.train_stages = 3
    return opt


def train_model(args):
    """Training model based on args.src_img_path"""
    opt = get_default_args()
    opt.gpu = args.gpu
    opt.not_cuda = args.not_cuda
    opt.train_mode = 'harmonization'
    opt.train_stages = 3
    opt.min_size = args.min_size
    opt.lrelu_alpha = 0.3
    opt.niter = 1000
    opt.batch_norm = True
    opt.input_name = args.src_img_path
    opt.naive_img = args.naive_img_path
    train.main(opt)
    print("Training done!")
    if not args.no_finetune:
        opt = get_default_args()
        opt.gpu = args.gpu
        opt.not_cuda = args.not_cuda
        opt.train_mode = "harmonization"
        opt.min_size = args.min_size
        opt.input_name = args.src_img_path
        opt.naive_img = args.naive_img_path
        opt.fine_tune = True
        args.fine_tune = False
        model_dir = generate_dir2save(args)
        opt.model_dir = model_dir

        args.fine_tune = True
        dir2save = generate_dir2save(args)
        if os.path.exists(dir2save):
            shutil.rmtree(dir2save)

        train.main(opt)
        print("Finetuning done!")


def harmonize(args):
    src_img_name = args.src_img_path.split("/")[-1][:-4]
    args.fine_tune = not args.no_finetune
    model_dir = generate_dir2save(args)
    opt = get_default_args()
    opt.fine_tune = args.fine_tune
    opt.gpu = args.gpu
    opt.not_cuda = args.not_cuda
    opt.min_size = args.min_size
    opt.model_dir = model_dir
    print(opt.model_dir)
    opt.naive_img = args.naive_img_path
    eval.main(opt)
    print("Harmonizing done!")
    return model_dir, src_img_name


def get_img(args, model_dir, name):
    img_dir = os.path.join(model_dir, "Evaluation/harmonized_w_mask.jpg")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(model_dir, "Evaluation/harmonized_wo_mask.jpg")
    # write_html(args, "Done!", "Output image:  ", output_img_dir)
    return img_dir
