from baiduAPI import *
import cv2


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


def get_coordinate(y_range, x_range):
    # 接受传入的x，y坐标，y:[0, y_range], x: [0, x_range]
    y = int(y_range/2)
    x = int(x_range/2)
    return y, x


def get_ratio():
    # 接受传入的缩放比例
    return 0.5


def get_naive(human_img, background_img, y, x):

    human_h, human_w = len(human_img), len(human_img[0])
    h, w = len(background_img), len(background_img[0])
    # print(x,  w, human_w)
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
        human_img = human_img[:, h + human_w - x - (x2-x1):w + human_w - x, :]
    else:
        x1 = max(x - human_w, 0)
        x2 = x
        human_img = human_img[:, -(x2-x1):, :]

    not_trans_indices = human_img[:, :, 3] > 250
    # print(y1, y2, x1, x2, background_img.shape,
    #   human_img.shape, not_trans_indices.shape)
    background_img[y1:y2, x1:x2,
                   :][not_trans_indices] = human_img[:, :, :3][not_trans_indices]
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
    mask_img = mask_img[top:bottom, left:right]
    h, w = len(background_img), len(background_img[0])

    size_ratio = get_ratio()
    human_img = cv2.resize(human_img, (0, 0), fx=size_ratio,
                           fy=size_ratio, interpolation=cv2.INTER_CUBIC)
    mask_img = cv2.resize(mask_img, (0, 0), fx=size_ratio,
                          fy=size_ratio, interpolation=cv2.INTER_CUBIC)
    human_h, human_w = len(human_img), len(human_img[0])

    # 接口：接受传入的x，y坐标，y:[0, h + human_h], x: [0, w + human_w]
    y, x = get_coordinate(y_range=h + human_h, x_range=w + human_w)

    naive_img = get_naive(human_img, background_img, y, x)
    naive_mask_img = get_naive(mask_img, empty_background, y, x)

    naive_img_path = "Images/tmp/naive_" + os.path.basename(
        human_img_path)[:-4] + "_" + os.path.basename(background_img_path)
    naive_mask_img_path = "Images/tmp/naive_" + os.path.basename(
        human_img_path)[:-4] + "_" + os.path.basename(background_img_path)[:-4] + "_mask.jpg"
    cv2.imwrite(naive_img_path, naive_img)
    cv2.imwrite(naive_mask_img_path, naive_mask_img)
    # print(naive_img_path)
    return naive_img_path
