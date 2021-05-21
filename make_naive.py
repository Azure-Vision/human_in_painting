from baiduAPI import *
import cv2
def segment_human(human_img_path):
    output_dir = os.path.dirname(human_img_path) + "/human_" + os.path.basename(human_img_path)
    segmentationAPI(img_name = human_img_path, foreground_name = output_dir)
    return output_dir
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
    return image[top:bottom, left:right]
def get_naive(human_img, background_img):
    # TODO: @szm
    y1, y2 = background_img.shape[0] - human_img.shape[0], background_img.shape[0]
    x2 = background_img.shape[1]
    x1 = x2 - human_img.shape[1]
    not_trans_indices = human_img[:, :, 3] > 250
    background_img[y1:y2, x1:x2, :][not_trans_indices] = human_img[:, :, :3][not_trans_indices]
    # print(background_img)
    return background_img

def get_naive_image(human_img_path, background_img_path):
    human_img = cv2.imread(human_img_path, cv2.IMREAD_UNCHANGED)
    background_img = cv2.imread(background_img_path, cv2.IMREAD_UNCHANGED)
    # print(human_img) [[[221 209 203   0]
    human_img = clear_border(human_img)
    naive_img = get_naive(human_img, background_img)
    naive_img_path = os.path.dirname(background_img_path) + "/naive_" + os.path.basename(human_img_path)[:-4] + "_" + os.path.basename(background_img_path)
    cv2.imwrite(naive_img_path, naive_img)
    print(naive_img_path)
    return naive_img_path