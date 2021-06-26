import datetime
import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree, rmtree
import glob
import time
import random
import torch

from Con_Sin_GAN.ConSinGAN.config import get_arguments
import Con_Sin_GAN.ConSinGAN.functions as functions
from Con_Sin_GAN.ConSinGAN.training_generation import *
from Con_Sin_GAN.ConSinGAN.training_harmonization_editing import *


def get_scale_factor(opt):
    opt.scale_factor = 1.0
    num_scales = math.ceil((math.log(math.pow(
        opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    opt.scale_factor_init = opt.scale_factor
    if opt.num_training_scales > 0:
        while num_scales > opt.num_training_scales:
            opt.scale_factor_init = opt.scale_factor_init - 0.01
            num_scales = math.ceil((math.log(math.pow(
                opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    return opt.scale_factor_init


def main(opt):

    opt = functions.post_config(opt)

    if opt.fine_tune:
        _gpu = opt.gpu
        _model_dir = opt.model_dir
        _timestamp = opt.timestamp
        _naive_img = opt.naive_img
        _niter = opt.niter

        opt = functions.load_config(opt)

        opt.gpu = _gpu
        opt.model_dir = _model_dir
        opt.start_scale = opt.train_stages - 1
        opt.timestamp = _timestamp
        opt.fine_tune = True
        opt.naive_img = _naive_img
        opt.niter = _niter

    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    dir2save = functions.generate_dir2save(opt)

    if os.path.exists(os.path.join(dir2save, "G.pth")):
        print('Trained model already exist: {}'.format(dir2save))
    else:
        # create log dir
        if os.path.exists(os.path.join(dir2save)):
            try:
                rmtree(dir2save)
                print(f"Directory {dir2save} is cleared.")
            except Exception:
                print(f"Directory {dir2save} is failed to be cleared.")
                pass
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        # save hyperparameters and code files
        with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
            for o in opt.__dict__:
                f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
        current_path = os.path.dirname(os.path.abspath(__file__))
        for py_file in glob.glob(osp.join(current_path, "*.py")):
            copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
        copytree(osp.join(current_path, "ConSinGAN"),
                 osp.join(dir2save, "ConSinGAN"))

        # train model
        print("Training model ({})".format(dir2save))
        start = time.time()
        train(opt)
        end = time.time()
        elapsed_time = end - start
        print("Time for training: {} seconds".format(elapsed_time))


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument(
        '--input_name', help='input image name for training', required=True)
    parser.add_argument(
        '--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget',
                                 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")
    parser.add_argument('--lr_scale', type=float,
                        help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int,
                        help='how many stages to use for training', default=6)

    parser.add_argument('--fine_tune', action='store_true',
                        help='whether to fine tune on a given image', default=0)
    parser.add_argument(
        '--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")

    opt = parser.parse_args()
    print(opt)
    main(opt)
