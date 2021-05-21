import os
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage import io
def write_html(args, title = "Put Me Into The Paintings", content = "", img = None):
    html_file = open("index.html", "w")
    img_line = ""
    if img:
        img_line = """<img src="{}" />""".format(os.path.join("Con_Sin_GAN", img))
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
def train_model(args):
    write_html(args, "Training...", "Training model based on " + args.src_img_path)
    command = "cd Con_Sin_GAN/\n python main_train.py --gpu " + str(args.gpu) + " --train_mode harmonization --train_stages 3 --min_size " + str(args.min_size) + " --lrelu_alpha 0.3 --niter 1000 --batch_norm --input_name " + args.src_img_path
    print(command)
    os.system(command)
    if args.finetune:
        src_img_name = args.src_img_path.split("/")[-1][:-4]
        model_dir = "TrainedModels/" + src_img_name + "/"
        latest_dir = sorted(os.listdir("Con_Sin_GAN/" + model_dir))[-1]
        model_dir += latest_dir
        command = "cd Con_Sin_GAN/\n python main_train.py --gpu " + str(args.gpu) + " --train_mode harmonization --input_name " + args.src_img_path + " --naive_img " + args.naive_img_path + " --fine_tune --model_dir " + model_dir
        os.system(command)
def harmoize(args):
    write_html(args, "Harmonizing...", "Harmonizing " + args.naive_img_path + " using trained model.")
    src_img_name = args.src_img_path.split("/")[-1][:-4]
    model_dir = "TrainedModels/" + src_img_name + "/"
    latest_dir = sorted(os.listdir("Con_Sin_GAN/" + model_dir))[-1]
    model_dir += latest_dir
    os.system("cd Con_Sin_GAN/\n python evaluate_model.py --gpu " + str(args.gpu) + " --model_dir " + model_dir + "/ --naive_img " + args.naive_img_path)
    return model_dir, src_img_name

def get_img(args, model_dir, name):
    img_dir = os.path.join(model_dir, "Evaluation/harmonized_w_mask.jpg")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(model_dir, "Evaluation/harmonized_wo_mask.jpg")
    output_img_dir = os.path.join(args.output_dir, name + ".jpg")
    cp_command = "cd Con_Sin_GAN/\n cp " + img_dir + " " + output_img_dir
    os.system(cp_command)
    write_html(args, "Done!", "Output image:  ", output_img_dir)



