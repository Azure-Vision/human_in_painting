from baiduAPI import *
import streamlit as st
import cv2
from auto_localization import auto_local


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def segment_human(human_img_path):
    output_dir = "Images/tmp/human_" + os.path.basename(human_img_path)
    mask_dir = "Images/tmp/mask_" + os.path.basename(human_img_path)
    segmentationAPI(img_name=human_img_path,
                    foreground_name=output_dir, labelmap_name=mask_dir)
    return output_dir, mask_dir


def clear_border(image):
    y = set()
    x = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, -1] != 0:
                y.add(i)
                x.add(j)
    top = min(y)
    bottom = max(y)
    left = min(x)
    right = max(x)
    return image[top:bottom, left:right], top, bottom, left, right


def get_coordinate(human_img, background_img):
    # 接受传入的x，y坐标，y:[0, y_range], x: [0, x_range]
    # y=0即人像底部贴着背景顶部，y=y_range即人像顶部贴着背景底部
    # x=0即人像右部贴着背景左部，x=x_range即人像左部贴着背景右部
    bottom_local = st.sidebar.checkbox(
        "Locality search at the bottom of painting image", True)
    best_y, best_x, best_scale = auto_local(
        background_img, human_img, False, bottom_local)
    y_range = len(background_img) + int(best_scale * len(human_img))
    x_range = len(background_img[0]) + int(best_scale * len(human_img[0]))
    x = st.sidebar.slider("x coordinate -> right", min_value=0,
                          max_value=x_range, value=best_x)
    y = st.sidebar.slider("y coordinate -> down", min_value=0,
                          max_value=y_range, value=best_y)
    r = st.sidebar.slider("scale ratio", min_value=0.01,
                          max_value=2.0, value=best_scale, step=0.01)
    return y, x, r


def get_angle():
    angle = st.sidebar.slider('angle -> clockwise', 0, 359, 0, 1)
    return angle


def get_naive(human_img, background_img, y, x):

    human_h, human_w = len(human_img), len(human_img[0])
    h, w = len(background_img), len(background_img[0])
    if y < human_h:
        y1 = 0
        y2 = min(y, h)
        human_img = human_img[human_h - y: human_h - y + y2, :, :]
    elif y > h:
        y1 = max(y - human_h, 0)
        y2 = h
        human_img = human_img[h + human_h - y - (y2-y1):h + human_h - y, :, :]
    else:
        y1 = max(y - human_h, 0)
        y2 = y
        human_img = human_img[-(y2-y1):, :, :]

    if x < human_w:
        x1 = 0
        x2 = min(x, w)
        human_img = human_img[:, human_w - x: human_w - x + x2, :]
    elif x > w:
        x1 = max(x - human_w, 0)
        x2 = w
        human_img = human_img[:, w + human_w - x - (x2-x1):w + human_w - x, :]
    else:
        x1 = max(x - human_w, 0)
        x2 = x
        human_img = human_img[:, -(x2-x1):, :]

    not_trans_indices = human_img[:, :, 3] > 250
    # print(y1, y2, x1, x2, background_img.shape,
    #   human_img.shape, not_trans_indices.shape)
    background_img[y1:y2, x1:x2,
                   :3][not_trans_indices] = human_img[:, :, :3][not_trans_indices]
    return background_img


def get_naive_and_mask(human_img_path, mask_img_path,  background_img_path):
    human_img = cv2.imread(human_img_path, cv2.IMREAD_UNCHANGED)
    # mask_img = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
    not_trans_indices = human_img[:, :, 3] > 250
    mask_img = np.zeros_like(human_img)
    mask_img[not_trans_indices] = 255
    background_img = cv2.imread(background_img_path, cv2.IMREAD_UNCHANGED)
    empty_background = np.zeros_like(background_img)

    human_img, top, bottom, left, right = clear_border(human_img)
    cv2.imwrite("Images/tmp/cleaned_human.jpg", human_img)
    mask_img = mask_img[top:bottom, left:right]
    h, w = len(background_img), len(background_img[0])

    # 接口：接受传入的x，y坐标，y:[0, h + human_h], x: [0, w + human_w]
    y, x, size_ratio = get_coordinate(
        human_img=human_img, background_img=background_img)
    angle = get_angle()
    human_img = rotate_bound(cv2.resize(
        human_img.copy(), (0, 0), fx=size_ratio, fy=size_ratio), angle)
    mask_img = rotate_bound(cv2.resize(
        mask_img.copy(), (0, 0), fx=size_ratio, fy=size_ratio), angle)
    human_h, human_w = len(human_img), len(human_img[0])

    naive_img = get_naive(human_img, background_img, y, x)
    naive_mask_img = get_naive(mask_img, empty_background, y, x)

    naive_img_path = "Images/tmp/naive_" + os.path.basename(
        human_img_path)[:-4] + "_" + os.path.basename(background_img_path)
    naive_mask_img_path = "Images/tmp/naive_" + os.path.basename(
        human_img_path)[:-4] + "_" + os.path.basename(background_img_path)[:-4] + "_mask.jpg"
    cv2.imwrite(naive_img_path, naive_img)
    cv2.imwrite(naive_mask_img_path, naive_mask_img)
    st.image(naive_img_path, "composite image", 300)
    # st.image(naive_mask_img_path, "composite image mask", 300)

    # print(naive_img_path)
    return naive_img_path
