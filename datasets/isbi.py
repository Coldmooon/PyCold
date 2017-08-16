from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageSequence:
    def __init__(self, im):
        self.im = im
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError # end of sequence

train_data_file = '/media/coldmoon/ExtremePro960G/Datasets/isbi2012seg/train-volume.tif'
train_labels_file = '/media/coldmoon/ExtremePro960G/Datasets/isbi2012seg/train-labels.tif'

imgs = Image.open(train_data_file)
labels = Image.open(train_labels_file)

# for i in range(30):
#     print(imgs.tell())
#     out = imgs.resize((256, 256))
#     print(out.size)
#     out.show()
#     imgs.show()
#     print(imgs.size)
#     labels.show()
#     imgs.seek(imgs.tell()+1)
#     labels.seek(labels.tell() + 1)


canvas = [47, 47]
size = int(imgs.size[0]/2)
dim_x = size - canvas[0] + 1
dim_y = size - canvas[1] + 1

train_dataset = np.zeros((dim_x * dim_y * 25, 1, canvas[0], canvas[1]), dtype=np.uint8)
test_dataset  = np.zeros((dim_x * dim_y * 5,  1, canvas[0], canvas[1]), dtype=np.uint8)

train_labels = np.zeros((dim_x * dim_y * 25), dtype=np.uint8)
test_labels =  np.zeros((dim_x * dim_y * 5), dtype=np.uint8)

stride = 1
train_count = 0
test_count = 0
for frame in ImageSequence(imgs):
    idx = frame.tell()
    labels.seek(idx)
    print('processing the', idx, 'th image ... and ', labels.tell(), 'th label')
    im = np.array(frame.resize((size, size)))
    # zero_padded_im = np.pad(im, pad_width=((23, 23), (23, 23)), mode='constant', constant_values=0)
    label = np.array(labels.resize((size, size)))

    for x in range(dim_x):
        # print('coor: x: [', x, ':', x + canvas[0], '];')
        for y in range(dim_y):
            # print('y: [', y, ':', y + canvas[1], '].')
            patch = im[x : x + canvas[0], y : y + canvas[1]]
            # padded_patch = zero_padded_im[x : x + canvas[0], y : y + canvas[1]]
            # print('label coor: [', x + canvas[0]/2,';', y + canvas[1]/2,']')
            patch_label = label[x + int(canvas[0]/2), y + int(canvas[1]/2)]
            if idx < 25:
                train_dataset[train_count, 0, :, :] = patch
                train_labels[train_count] = patch_label
                train_count += 1
            else:
                test_dataset[test_count, 0, :, :] = patch
                test_labels[test_count] = patch_label
                test_count += 1


    print('cut out ', train_count, 'training samples and', test_count, 'test samples ...')


# data balance
n_train_class_white = np.sum(train_labels == 255)
train_class_white_index = np.where(train_labels == 255)[0]
train_picked_white_index_index = np.random.choice(train_class_white_index.shape[0], 200000, replace=False)
train_selected_white_index = train_class_white_index[train_picked_white_index_index]

n_train_class_black = np.sum(train_labels == 0)
train_class_black_index = np.where(train_labels == 0)[0]
train_picked_black_index_index = np.random.choice(train_class_black_index.shape[0], 200000, replace=False)
train_selected_black_index = train_class_black_index[train_picked_black_index_index]

train_selected_index = np.append(train_selected_white_index, train_selected_black_index)
np.random.shuffle(train_selected_index)

train_data_selected = train_dataset[train_selected_index,:,:,:]
train_labels_selected = train_labels[train_selected_index]

# for torch
np.save('train_dataset.npy', train_data_selected)
np.save('train_labels.npy', train_labels_selected)
np.save('test_dataset.npy', test_dataset)
np.save('test_labels.npy', test_labels)
