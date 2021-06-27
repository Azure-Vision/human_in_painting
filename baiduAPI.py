import requests
import base64
import os
import cv2
import numpy as np


# need image name as input, other names for output
def segmentationAPI(img_name='test.png', labelmap_name=None, foreground_name=None, scoremap_name=None):

    # 连接API
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    API_key = 'GcyS0DBmHlPdgI4GV51kAKYR'
    secret_key = 'kPhNRD7O4Aq1xqUjrWYLQj5NgxqSNfYF'

    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + \
        API_key + '&client_secret=' + secret_key
    # print('host', host)
    connect_response = requests.get(host)
    if connect_response:
        # print(connect_response.json())
        # print('connect!')
        pass

    access_token = connect_response.json()['access_token']

    '''
    人像分割 https://ai.baidu.com/ai-doc/BODY/Fk3cpyxua
    '''

    # 调用人像分割
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_seg"
    # 二进制方式打开图片文件
    f = open(img_name, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img}
    access_token = access_token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        # print(response.json().keys())
        pass

    # 保存labelmap：分割结果图片，base64编码之后的二值图像，需二次处理方能查看分割效果
    if labelmap_name is not None:
        img = cv2.imread(img_name)
        width = img.shape[1]
        height = img.shape[0]
        labelmap = base64.b64decode(
            response.json()['labelmap'])  # res为通过接口获取的返回json
        nparr = np.frombuffer(labelmap, np.uint8)
        labelimg = cv2.imdecode(nparr, 1)
        # width, height为图片原始宽、高
        labelimg = cv2.resize(labelimg, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        im_new = np.where(labelimg == 1, 255, labelimg)
        cv2.imwrite(labelmap_name, im_new)

    # 保存foreground：分割后的人像前景抠图，透明背景，Base64编码后的png格式图片，不用进行二次处理，直接解码保存图片即可。将置信度大于0.5的像素抠出来，并通过image matting技术消除锯齿
    if foreground_name is not None:
        foreground_img = base64.b64decode(response.json()['foreground'])
        file = open(foreground_name, 'wb')
        file.write(foreground_img)
        file.close()

    # 保存scoremap：分割后人像前景的scoremap，归一到0-255，不用进行二次处理，直接解码保存图片即可。Base64编码后的灰度图文件，图片中每个像素点的灰度值 = 置信度 * 255，置信度为原图对应像素点位于人体轮廓内的置信度，取值范围[0, 1]
    if scoremap_name is not None:
        scoremap_img = base64.b64decode(response.json()['scoremap'])
        file = open(scoremap_name, 'wb')
        file.write(scoremap_img)
        file.close()


# need image name as input
def getLabelMap(img_name='test.png'):

    # 连接API
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    API_key = 'GcyS0DBmHlPdgI4GV51kAKYR'
    secret_key = 'kPhNRD7O4Aq1xqUjrWYLQj5NgxqSNfYF'

    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + \
        API_key + '&client_secret=' + secret_key
    print('host', host)
    connect_response = requests.get(host)
    if connect_response:
        # print(connect_response.json())
        print('connect!')

    access_token = connect_response.json()['access_token']

    '''
    人像分割 https://ai.baidu.com/ai-doc/BODY/Fk3cpyxua
    '''

    # 调用人像分割
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_seg"
    # 二进制方式打开图片文件
    f = open(img_name, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img}
    access_token = access_token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print(response.json().keys())

    img = cv2.imread(img_name)
    width = img.shape[1]
    height = img.shape[0]
    labelmap = base64.b64decode(
        response.json()['labelmap'])  # res为通过接口获取的返回json
    nparr = np.frombuffer(labelmap, np.uint8)
    labelimg = cv2.imdecode(nparr, 1)
    # width, height为图片原始宽、高
    labelimg = cv2.resize(labelimg, (width, height),
                          interpolation=cv2.INTER_NEAREST)
    im_new = np.where(labelimg == 1, 255, labelimg)
    return im_new


if __name__ == "__main__":
    segmentationAPI(img_name='test.png', labelmap_name='test_labelmap.png',
                    foreground_name='test_foreground.png', scoremap_name='test_scoremap.png')
