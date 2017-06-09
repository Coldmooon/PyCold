from interpolation import *
import numpy as np
from utils import img_to_array, array_to_img

DIMS = (400, 400)
CAT1 = 'cat1.jpg'
CAT2 = 'cat2.jpg'
data_path = './data/'

# load 4 cat images
img1 = img_to_array(data_path + CAT1, DIMS)
img2 = img_to_array(data_path + CAT2, DIMS, view=True)

# concat into tensor of shape (2, 400, 400, 3)
input_img = np.concatenate([img1, img2], axis=0)

# dimension sanity check
print("Input Img Shape: {}".format(input_img.shape))

M = np.array([[2, 0, 0],
              [0, 2, 0]])

im = affine_transformer(input_img, M)

print(im.shape)

# view the 2nd image
im = array_to_img(im[-1])
im.show()

print('done!')