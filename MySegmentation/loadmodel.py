import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import DsDataset
#from evaluation import evaluate
#from model import Net
import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import IPython
import torchvision
from torchvision import datasets, transforms
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate
#from efficientnet_pytorch import EfficientNet  # EfficientNet的使用需要倒入的库
from PIL import ImageFile
import cv2
import segmentation_models_pytorch as smp


ImageFile.LOAD_TRUNCATED_IMAGES = True

curdir = os.path.dirname(__file__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=50, type=int)
    parser.add_argument("--save-interval", default=5, type=int)
    parser.add_argument("--step-interval", default=10, type=int)
    parser.add_argument("--step-save", default=1000, type=int)
    parser.add_argument("--evaluate-step", default=100, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "/mnt/sda1/songzimeng/cv_project/models/"))
    parser.add_argument("--total-updates", default=50000, type=int)
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--class_num", default=2, type=int)
    parser.add_argument("--cuda_num", default=3, type=int)
    parser.add_argument("--val_set", default=0, type=int)
    parser.add_argument("--path_label", default='/mnt/sda1/songzimeng/cv_project/label/')
    parser.add_argument("--path_img", default='/mnt/sda1/songzimeng/cv_project/img/')
    parser.add_argument("--load_path", default='/models/best.pt')
    parser.add_argument("--encoder-type", default='resnet34')
    parser.add_argument("--model-type", default='FPN')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_num)
    print(device)
    if torch.cuda.is_available():
        print('device: ', torch.cuda.current_device())
    model = torch.load(args.load_path, map_location={'cuda:0': 'cuda:' + str(args.cuda_num), 'cuda:1': 'cuda:' + str(args.cuda_num), 'cuda:2': 'cuda:' + str(args.cuda_num), 'cuda:3': 'cuda:' + str(args.cuda_num)}).to(
        device)


    model.eval()
    with torch.no_grad():
        img_path = './pics/test.png'
        height = 800
        width = 1280
        photo = cv2.imread(img_path)
        #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        tmp = np.zeros((height, width, 3))
        h = min(photo.shape[0], height)
        w = min(photo.shape[1], width)
        tmp[:h, :w, :] = photo[:h, :w, :]
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = data_transform(tmp)
        #IPython.embed()
        #os._exit(0)

        outputs = model(torch.tensor(tmp).unsqueeze(0).permute(0,3,1,2).to(device).float())
        outputs = outputs.squeeze().permute(1, 2, 0)
        outputs = torch.argmax(outputs.reshape(-1, args.class_num), dim=-1)
        outputs = outputs.reshape(height, width)

        out_pic = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                if outputs[i][j] == 0:
                    out_pic[i][j] = np.array([255,255,255])
                else:
                    out_pic[i][j] = photo[i][j]

        cv2.imwrite('./pics/Mytest_labelmap.png', out_pic)

        #IPython.embed()
        #os._exit(0)
