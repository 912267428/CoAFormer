import os
import PIL.Image as Image
import numpy as np

if __name__ == '__main__':
    root_path = '/home/pc/label/correct/Internet'
    cla_list = os.listdir(root_path)

    for cla in cla_list:
        image_path = os.path.join(root_path, cla)
        label_path = os.path.join(image_path, 'label')
        mask_path = os.path.join(label_path, 'mask')
        ground_path = os.path.join(mask_path, 'Groundtruth')
        if not os.path.exists(ground_path):
            os.mkdir(ground_path)

        mask_list = os.listdir(mask_path)
        mask_list.remove('Groundtruth')
        for mask_name in mask_list:
            mask = Image.open(os.path.join(mask_path, mask_name))
            mask = np.array(mask)
            mask[mask == 255] = 0

            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            # mask.show()
            mask.save(os.path.join(ground_path, mask_name))