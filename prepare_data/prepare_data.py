import os
import time
import h5py
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import skimage.io as io
from skimage.transform import resize

import matplotlib.pyplot as plt

input_dir = r'E:\50_plex\tif\pipeline2\unmixed'
bbxs_file = r'E:\50_plex\tif\pipeline2\detection_results\bbxs_detection.txt'
channelInfo_file = r'E:\50_plex\scripts\channel_info.csv'
parallel = True

margin = 5
image_size = (50, 50)
topN = 5000

biomarkers = ['DAPI', 'Histones', 'NeuN', 'S100', 'Olig2', 'Iba1', 'RECA1']


def zero_pad(image, dim):
    """
    pad zeros to the image in the first and second dimension
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                 (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                 (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def to_square(image):
    """
    pad zeros to the image to make it square
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    dim = max(image.shape[:2])
    if image.shape[0] >= image.shape[1]:
        pad_width = ((0, 0),
                     (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                     (0, 0))
    else:
        pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                     (0, 0),
                     (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def main():

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default


    # read channel info table
    assert os.path.isfile(channelInfo_file), '{} not found!'.format(channelInfo_file)
    chInfo = pd.read_csv(channelInfo_file, sep=',')

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep='\t')
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values

    # get bounding boxes in the center of brain
    bbxs = bbxs[(bbxs[:, 0] >= 8000) & (bbxs[:, 2] <= 34000) &
                (bbxs[:, 1] >= 4000) & (bbxs[:, 3] <= 24000)]

    # shuffle the bounding boxes
    permutation = np.random.permutation(bbxs.shape[0])
    bbxs = bbxs[permutation, :]


    # get channels full address from channel info table
    channel_names = [chInfo.loc[chInfo['Biomarker'] == bioM]['Channel'].values[0] for bioM in biomarkers]
    channel_names = [os.path.join(input_dir, ch) for ch in channel_names]

    # get image collection
    im_coll = io.imread_collection(channel_names, plugin='tifffile')
    images = io.concatenate_images(im_coll)
    images = np.moveaxis(images, 0, -1)     # put channel as last dimension

    # from utils import bbxs_image
    # bbxs_image('all.tif', bbxs, images[:, :, 0].shape[::-1])

    # crop image (with extra margin) to get each sample (cell)
    def get_crop(image, bbx, margin=0):
        return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]


    ################### GENERATE LABELS ###############################
    # calculate the intensity of each channel
    intensities = np.array([np.mean(get_crop(images, bbx), axis=(0, 1)) for bbx in bbxs])
    intensities = intensities[:, 2:]         # we don't need DAPI and Histones for classification
    # find top N cells with highest intensity
    top_cells_each = [(-intensities[:, i]).argsort()[:topN] for i in range(intensities.shape[1])]
    top_cells = np.unique(np.array(top_cells_each).flatten())

    intensities = intensities[top_cells, :]
    labels = (intensities == intensities.max(axis=1)[:, None]).astype(int)
    print(np.sum(labels, axis=0))

    ################### GENERATE IMAGES ###############################
    # update bounding boxes to keep the desired cells
    bbxs = bbxs[top_cells, :]
    # get the crops
    cells = [get_crop(images, bbx, margin=margin) for bbx in bbxs]
    # del images

    # zero pad to the maximum dim
    max_dim = max((max([cell.shape[:2] for cell in cells]))) # find maximum in each dimension
    if parallel:
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_cells = pool.map(zero_pad_x, cells)
    else:
        new_cells = [zero_pad(cell, max_dim) for cell in cells]

    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=image_size, mode='constant', preserve_range=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_new_cells = pool.map(resize_x, new_cells)
    else:
        new_new_cells = [resize(cell, image_size, mode='constant', preserve_range=True) for cell in new_cells]

    # visualize
    # id = 14000
    # fig = plt.figure(figsize=(10,2))
    # for i in range(7):
    #     plt.subplot(1, 7, i + 1)
    #     plt.imshow(new_new_cells[id][:, :, i], cmap='gray', vmin=0, vmax=np.max(cells[id]))
    #     plt.title(biomarkers[i])
    # plt.tight_layout()
    # fig.suptitle('LABEL = {}'.format(biomarkers[np.argmax(labels[id])+2]))
    # plt.show()

    cells = np.array(new_new_cells)

    from utils import bbxs_image
    bbxs_image(biomarkers[2] + '.tif', bbxs[labels[:, 0] == 1, :], images[:, :, 3].shape[::-1], color='red')
    bbxs_image(biomarkers[3] + '.tif', bbxs[labels[:, 1] == 1, :], images[:, :, 0].shape[::-1], color='red')
    bbxs_image(biomarkers[4] + '.tif', bbxs[labels[:, 2] == 1, :], images[:, :, 0].shape[::-1], color='red')
    bbxs_image(biomarkers[5] + '.tif', bbxs[labels[:, 3] == 1, :], images[:, :, 0].shape[::-1], color='red')
    bbxs_image(biomarkers[6] + '.tif', bbxs[labels[:, 4] == 1, :], images[:, :, 0].shape[::-1], color='red')


    with h5py.File('data.h5', 'w') as f:
        f.create_dataset('X', data=cells)
        f.create_dataset('Y', data=labels)

if __name__ == '__main__':
    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
