import torch
import numpy as np
from utils import img_to_array, array_to_img

def affine_grid_generator(height, width, M):
	"""
	This function returns a sampling grid, which when
	used with the bilinear sampler on the input img,
	will create an output img that is an affine
	transformation of the input.

	Input
	-----
	- M: affine transform matrices of shape (num_batch, 2, 3).
	  For each image in the batch, we have 6 parameters of
	  the form (2x3) that define the affine transformation T.

	Returns
	-------
	- normalized gird (-1, 1) of shape (num_batch, H, W, 2).
	  The 4th dimension has 2 components: (x, y) which are the
	  sampling points of the original image for each point in the
	  target image.
	"""

	# create normalized 2D grid
	x = torch.linspace(-1, 1, width)
	y = torch.linspace(-1, 1, height)

	dim_x, = x.size()
	dim_y, = y.size()

	x = x.expand(dim_y,dim_x).contiguous()
	y = y[:,None].expand(dim_x,dim_y).contiguous()

	dimensions = torch.prod(torch.IntTensor([*x.size()]))

	# reshape to (xt, yt, 1)
	x = x.view(dimensions)
	y = y.view(dimensions)
	ones = torch.ones(dimensions)

	sampling_grid = torch.stack([x,y,ones], 0)
	# homogeneous coordinates

	batch_grids = torch.mm(M, sampling_grid)
	batch_grids = batch_grids.view(2, height, width).permute(1,2,0)
	# batch_grids = batch_grids.expand(2, *batch_grids.size())

	return batch_grids


def bilinear_sampler(input_img, x, y):

	B, H, W, C = input_img.shape

	# rescale x and y to [0, W/H]
	x = ((x + 1.) * W) * 0.5
	y = ((y + 1.) * H) * 0.5

	# grab 4 nearest corner points for each (x_i, y_i)
	x0 = torch.floor(x).int().numpy()
	x1 = x0 + 1
	y0 = torch.floor(y).int().numpy()
	y1 = y0 + 1

	# make sure it's inside img range [0, H] or [0, W]
	x0[x0 < 0] = 0
	x0[x0 > W -1] = W - 1

	x1[x1 < 0] = 0
	x1[x1 > W - 1] = W - 1
	y0[y0 < 0] = 0
	y0[y0 > H - 1] = H - 1
	y1[y1 < 0] = 0
	y1[y1 > H -1] = H -1


	# look up pixel values at corner coords
	Ia = input_img[np.arange(B)[:,None,None], y0, x0]
	Ib = input_img[np.arange(B)[:,None,None], y1, x0]
	Ic = input_img[np.arange(B)[:,None,None], y0, x1]
	Id = input_img[np.arange(B)[:,None,None], y1, x1]

	# calculate deltas
	wa = (x1-x.numpy()) * (y1-y.numpy())
	wb = (x1-x.numpy()) * (y.numpy()-y0)
	wc = (x.numpy()-x0) * (y1-y.numpy())
	wd = (x.numpy()-x0) * (y.numpy()-y0)

	# add dimension for addition
	wa = np.expand_dims(wa, axis=3)
	wb = np.expand_dims(wb, axis=3)
	wc = np.expand_dims(wc, axis=3)
	wd = np.expand_dims(wd, axis=3)

	# compute output
	out = wa*Ia + wb*Ib + wc*Ic + wd*Id

	# out = torch.from_numpy(out)

	return out

def affine_transformer(input_image, out_H=None, out_W=None, rotation=None):
	# grab shape
	B, H, W, C = input_image.shape

	if out_H == None:
		out_H = H
	if out_W == None:
		out_W = W

	param1 = param5 = 1.
	param2, param3, param4, param6 = 0, 0, 0, 0

	if(rotation):
		param1, param5 = np.cos(rotation), np.cos(rotation)
		param2 = -np.sin(rotation)
		param4 = -param2

	# initialize theta to identity transform
	M = torch.FloatTensor([[param1, param2, param3],
				           [param4, param5, param6]])

	# repeat num_batch times
	# M = M.expand(B, *M.size())

	# get grids
	batch_grids = affine_grid_generator(out_H, out_W, M)

	x_s = batch_grids[:, :, 0:1].squeeze()
	y_s = batch_grids[:, :, 1:2].squeeze()

	transformed_image = bilinear_sampler(input_image, x_s, y_s)


	return transformed_image