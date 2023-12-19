import os
import shutil

root = './'
JPEG_path = root + 'JPEGImages'
MASK_path = root + 'GroundTruth'
use_path = root + 'UsedImage'

JPEG_list = os.listdir(JPEG_path)
MASK_list = os.listdir(MASK_path)

for mask in MASK_list:
    replace = mask.replace('.png', '.jpg')
    if replace in JPEG_list:
        shutil.copy(os.path.join(JPEG_path, replace), os.path.join(use_path, replace))