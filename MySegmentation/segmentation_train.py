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
    parser.add_argument("--encoder-type", default='resnet34')
    parser.add_argument("--model-type", default='FPN')
    args = parser.parse_args()
    return args


def train(args):
    print(args)
    args.save_dir = args.save_dir + 'val_' + str(args.val_set)
    args.save_dir += "_" + args.model_type + args.encoder_type + "_"
    args.save_dir = args.save_dir + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.save_dir, 'make!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_num)
    print(device)
    if torch.cuda.is_available():
        print('device: ', torch.cuda.current_device())

    path_label = args.path_label
    path_img = args.path_img

    city_list_train = []
    city_list_val = []
    for i in range(10):
        city_list_train.append('ds' + str(i+1))
    for i in range(10, 13):
        city_list_val.append('ds' + str(i + 1))
    print('train:', city_list_train)
    print('valid:', city_list_val)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = DsDataset(path_label, path_img, city_list_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set = DsDataset(path_label, path_img, city_list_val)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    #https://segmentation-modelspytorch.readthedocs.io/en/latest/
    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=args.class_num).to(device)
    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=args.class_num).to(device)
    if args.model_type == 'FPN':
        model = smp.FPN(args.encoder_type, classes=args.class_num, encoder_weights='imagenet').to(device)
    elif args.model_type == 'Unet':
        model = smp.Unet(args.encoder_type, classes=args.class_num, encoder_weights='imagenet').to(device)
    elif args.model_type == 'Linknet':
        model = smp.Linknet(args.encoder_type, classes=args.class_num, encoder_weights='imagenet').to(device)
    elif args.model_type == 'PSPNet':
        model = smp.PSPNet(args.encoder_type, classes=args.class_num, encoder_weights='imagenet').to(device)
    else:
        print('Wrong Model!')

    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-9)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

    # evaluate(model, valid_loader)

    global_step = 0
    best_MIoU = 0

    for epoch in range(args.num_epoch):
        print('epoch: ', epoch+1)

        losses = []
        IoU = []

        optimizer.zero_grad()
        for step, samples in enumerate(train_loader, 0):
            model.train()
            imgs, labels = samples['image'].to(device).float(), samples['label'].to(device)
            labels = torch.sum(labels, dim=-1)
            labels = torch.where(labels > 1, torch.ones_like(labels), labels)
            labels = torch.where(labels < 1, torch.zeros_like(labels), labels)

            #IPython.embed()
            #os._exit(0)

            if imgs.shape[0] == 1:
                continue
            outputs = model(imgs)

            loss = criterion(outputs.permute(0, 2, 3, 1).reshape(-1, args.class_num), labels.reshape(-1).long())
            loss.backward()
            losses.append(loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            with torch.no_grad():
                outputs = torch.argmax(outputs.permute(0, 2, 3, 1).reshape(-1, args.class_num), dim=-1)
                labels = labels.reshape(-1).long()
                #outputs[labels == 0] = 0
                step_IoU = []
                for class_id in range(args.class_num):

                    # get the position of this class
                    outputs_mask = (outputs == class_id)
                    labels_mask = (labels == class_id)

                    # calculate I and U
                    step_I = (outputs_mask * labels_mask).sum().item()
                    step_U = (outputs_mask + labels_mask).sum().item()
                    step_IoU.append(step_I/max(step_U, 1.0))
                IoU.append(np.mean(step_IoU))

            if (step + 1) % args.step_interval == 0:
               print('[epoch:%d, iter:%d] Loss: %.03f | MIoU: %.3f%% ' \
                      % (epoch + 1, step + 1, np.mean(losses), 100.0 * np.mean(IoU)))

            #IPython.embed()
            #os._exit(0)



        if (epoch + 1) % args.save_interval == 0 or (epoch in [0, 1, 2]):
            torch.save(model, args.save_dir + "/val_" + str(args.val_set) + "_epoch_{}.pt".format(epoch + 1))

        if optimizer.param_groups[0]['lr'] == 0:
            break

        with torch.no_grad():
            print('Evaluate')
            model.eval()

            eval_losses = []
            eval_IoU = []

            for eval_step, samples in enumerate(valid_loader, 0):

                imgs, labels = samples['image'].to(device).float(), samples['label'].to(device)
                labels = torch.sum(labels, dim=-1)
                labels = torch.where(labels > 1, torch.ones_like(labels), labels)
                labels = torch.where(labels < 1, torch.zeros_like(labels), labels)

                outputs = model(imgs)
                loss = criterion(outputs.permute(0, 2, 3, 1).reshape(-1, args.class_num), labels.reshape(-1).long())
                eval_losses.append(loss.item())

                outputs = torch.argmax(outputs.permute(0, 2, 3, 1).reshape(-1, args.class_num), dim=-1)
                labels = labels.reshape(-1).long()
                #outputs[labels == 0] = 0
                eval_step_IoU = []
                for class_id in range(args.class_num):
                    # get the position of this class
                    outputs_mask = (outputs == class_id)
                    labels_mask = (labels == class_id)
                    # calculate I and U
                    step_I = (outputs_mask * labels_mask).sum().item()
                    step_U = (outputs_mask + labels_mask).sum().item()
                    eval_step_IoU.append(step_I / max(step_U, 1.0))
                eval_IoU.append(np.mean(eval_step_IoU))

                if eval_step >= args.evaluate_step:
                    break


            mean_eval_IoU = np.mean(eval_IoU)
            print('[epoch:%d, evaluate] Loss: %.03f | MIoU: %.3f%% ' \
                          % (epoch + 1, np.mean(eval_losses), 100.0 * mean_eval_IoU))

            if mean_eval_IoU > best_MIoU:
                best_MIoU = mean_eval_IoU
                torch.save(model, args.save_dir + "/Best_val_" + str(args.val_set) + ".pt")
                print('Epoch {} achieves best!'.format(epoch + 1))


if __name__ == "__main__":
    args = get_args()
    train(args)