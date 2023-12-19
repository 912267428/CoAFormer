import os

import PIL.Image as Image
import numpy as np
import cv2 as cv

# path = './Data/MSRC/tree/2_1_s.bmp'
#
# mask = Image.open(path).convert('RGB')
# mask.show()
# numpy_n = np.array(mask)
# mask_cv = cv.imread(path)
# print(mask_cv == numpy_n)
# print(123)
Out_path = './Checkpoints/CoAFormer/train_byvoc_DCsample_fusion/'
history = np.load(os.path.join(Out_path, 'history.npy'), allow_pickle=True).item()

print(123)