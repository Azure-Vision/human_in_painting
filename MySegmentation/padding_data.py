import cv2
import os
import numpy as np
import IPython


label_path = '/mnt/sda1/songzimeng/cv_project/Supervisely_Person_Mask/'
destination_path = '/mnt/sda1/songzimeng/cv_project/label/'
img_path = '/mnt/sda1/songzimeng/cv_project/Supervisely_Person_Dataset/'
img_destination_path = '/mnt/sda1/songzimeng/cv_project/img/'


height = 800
width = 1280

for ds in os.listdir(label_path):
    ds_label_path = label_path + ds + '/'
    ds_des_path = destination_path + ds + '/'
    img_des_path = img_destination_path + ds + '/'
    if not os.path.exists(ds_des_path):
        os.mkdir(ds_des_path)
    if not os.path.exists(img_des_path):
        os.mkdir(img_des_path)
    for png_file in os.listdir(ds_label_path):
        label_file_path = ds_label_path + png_file
        des_file_path = ds_des_path + png_file

        photo = cv2.imread(label_file_path)
        tmp = np.zeros((height, width, 3))
        h = min(photo.shape[0], height)
        w = min(photo.shape[1], width)
        tmp[:h, :w, :] = photo[:h, :w, :]
        cv2.imwrite(des_file_path, tmp)

        photo = cv2.imread(img_path + ds + '/img/' + png_file)
        tmp = np.zeros((height, width, 3))
        h = min(photo.shape[0], height)
        w = min(photo.shape[1], width)
        tmp[:h, :w, :] = photo[:h, :w, :]
        cv2.imwrite(img_destination_path + ds + '/' + png_file, tmp)

    print(ds_label_path, ds_des_path)


