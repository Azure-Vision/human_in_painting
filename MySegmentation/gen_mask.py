import os
import numpy as np
import cv2
import tqdm
import supervisely_lib as sly
from matplotlib import pyplot as plt


def genMask(dataset_root_dir="/mnt/sda1/songzimeng/cv_project/Supervisely_Person_Dataset",
            output_dir="/mnt/sda1/songzimeng/cv_project/Supervisely_Person_Mask",
            label_color=[255, 255, 255]):
    ''' 生成Mask
        dataset_root_dir: 数据集根目录
        output_dir: 生成mask的保存目录
        label_color: mask 标注的颜色[R, G, B]
    '''
    project = sly.Project(dataset_root_dir, sly.OpenMode.READ)

    # 打印数据集信息
    print(f"Project name     : {project.name}")
    print(f"Project directory: {project.directory}")
    print(f"Total images     : {project.total_items}")
    print(f"Dataset names    : {project.datasets.keys()}")
    print()
    print(project.meta)

    pbar = tqdm.tqdm(total=project.total_items)
    for dataset in project:
        for item_name in dataset:
            # 更新进度条
            pbar.update(1)
            # 获取原始图像和标注文件路径
            item_paths = dataset.get_item_paths(item_name)
            # 加载注释文件
            ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
            # 创建一个用于渲染标注的3通道黑色画布，
            ann_render = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            # 渲染所有的标注（该数据集只有人）
            ann.draw(ann_render, color=label_color)
            # ann_render shape: (h, w, c), pixel: (R, G, B)
            # RGB -> BGR，用于Opencv
            ann_render = ann_render[..., ::-1]
            # mask 保存目录不存在，则创建
            save_dir = os.path.join(output_dir, dataset.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存mask
            mask_path = os.path.join(save_dir, item_name)
            cv2.imwrite(mask_path, ann_render)
    pbar.close()


if __name__ == '__main__':
    import fire

    fire.Fire(genMask, name='gen_mask')
    '''
    Usage:
    # 1. 使用默认配置
    python code/gen_mask.py
    # 2. 指定数据库目录和mask输出目录
    python gen_mask.py --dataset_root_dir ../person --output_dir ../person_mask
    '''

# 原文链接：https: // blog.csdn.net / SimleCat / article / details / 107022348