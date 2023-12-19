import os
import json
import numpy as np
import cv2
import PIL.Image as Image


def get_palette():
    origin = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
              (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
              (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
              (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
    palette = []
    for i in range(255):
        if i + 1 <= len(origin):
            palette.append(origin[i][0])
            palette.append(origin[i][1])
            palette.append(origin[i][2])
        else:
            palette.append(0)
            palette.append(0)
            palette.append(0)
    palette.append(224)
    palette.append(224)
    palette.append(192)

    return palette


def get_classinfo():
    class_info = [
        {'label': 'background', 'type': 'polygon', 'color': 0},
        {'label': 'airplane', 'type': 'polygon', 'color': 1},
        {'label': 'bicycle', 'type': 'polygon', 'color': 2},
        {'label': 'bird', 'type': 'polygon', 'color': 3},
        {'label': 'boat', 'type': 'polygon', 'color': 4},
        {'label': 'bottle', 'type': 'polygon', 'color': 5},
        {'label': 'bus', 'type': 'polygon', 'color': 6},
        {'label': 'car', 'type': 'polygon', 'color': 7},
        {'label': 'cat', 'type': 'polygon', 'color': 8},
        {'label': 'chair', 'type': 'polygon', 'color': 9},
        {'label': 'cow', 'type': 'polygon', 'color': 10},
        {'label': 'diningtable', 'type': 'polygon', 'color': 11},
        {'label': 'dog', 'type': 'polygon', 'color': 12},
        {'label': 'horse', 'type': 'polygon', 'color': 13},
        {'label': 'motorbike', 'type': 'polygon', 'color': 14},
        {'label': 'person', 'type': 'polygon', 'color': 15},
        {'label': 'pottedplant', 'type': 'polygon', 'color': 16},
        {'label': 'sheep', 'type': 'polygon', 'color': 17},
        {'label': 'sofa', 'type': 'polygon', 'color': 18},
        {'label': 'train', 'type': 'polygon', 'color': 19},
        {'label': 'tvmonitor', 'type': 'polygon', 'color': 20}
    ]

    return class_info


def Tomask(path, class_info, palette):
    labelme_json_path = path
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
    img_mask = np.zeros([labelme['imageHeight'], labelme['imageWidth']])

    for one_class in class_info:  # 按顺序遍历每一个类别
        for each in labelme['shapes']:  # 遍历所有标注，找到属于当前类别的标注
            if each['label'] == one_class['label']:
                if one_class['type'] == 'polygon':  # polygon 多段线标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（闭合区域）
                    img_mask = cv2.fillPoly(img_mask, points, color=one_class['color'])

                elif one_class['type'] == 'line' or one_class['type'] == 'linestrip':  # line 或者 linestrip 线段标注

                    # 获取点的坐标
                    points = [np.array(each['points'], dtype=np.int32).reshape((-1, 1, 2))]

                    # 在空白图上画 mask（非闭合区域）
                    img_mask = cv2.polylines(img_mask, points, isClosed=False, color=one_class['color'],
                                             thickness=one_class['thickness'])

                elif one_class['type'] == 'circle':  # circle 圆形标注
                    points = np.array(each['points'], dtype=np.int32)
                    center_x, center_y = points[0][0], points[0][1]  # 圆心点坐标
                    edge_x, edge_y = points[1][0], points[1][1]  # 圆周点坐标
                    radius = np.linalg.norm(np.array([center_x, center_y] - np.array([edge_x, edge_y]))).astype(
                        'uint32')  # 半径
                    img_mask = cv2.circle(img_mask, (center_x, center_y), radius, one_class['color'],
                                          one_class['thickness'])
                else:
                    print('未知标注类型', one_class['type'])

    mask_i = Image.fromarray(img_mask)
    mask_i = mask_i.convert("P")
    mask_i.putpalette(palette)

    return mask_i


if __name__ == '__main__':
    root_path = '/home/pc/label/correct/Internet'
    palette = get_palette()
    class_info = get_classinfo()
    cla_list = os.listdir(root_path)

    for cla in cla_list:
        image_path = os.path.join(root_path, cla)
        label_path = os.path.join(image_path, 'label')
        mask_path = os.path.join(label_path, 'mask')
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        label_list = os.listdir(label_path)
        label_list.remove('mask')
        for label in label_list:
            now_label_path = os.path.join(label_path, label)
            mask = Tomask(now_label_path, class_info, palette)
            mask.save(os.path.join(label_path, 'mask', label.replace('.json','.png')))