import os
import time
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

input_dir = r'E:\50_plex\tif\pipeline2\unmixed'
bbxs_file = r'E:\50_plex\tif\pipeline2\detection_results\bbxs_detection.txt'
channelInfo_file = r'E:\50_plex\scripts\channel_info.csv'

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
    # read channel info table
    assert os.path.isfile(channelInfo_file), '{} not found!'.format(channelInfo_file)
    chInfo = pd.read_csv(channelInfo_file, sep=',')

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep='\t')
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values


    # get channels full address from channel info table
    channel_names = [chInfo.loc[chInfo['Biomarker'] == bioM]['Channel'].values[0] for bioM in biomarkers]
    channel_names = [os.path.join(input_dir, ch) for ch in channel_names]

    # get image collection
    im_coll = io.imread_collection(channel_names, plugin='tifffile')
    images = io.concatenate_images(im_coll)
    images = np.moveaxis(images, 0, -1)     # put channel as last dimension

    # crop image (with extra margin) to get each sample (cell)
    margin = 5
    cells = [images[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :] for bbx in bbxs]
    # del images

    # zero pad to the maximum dim
    # find maximum in each dimension for zero-pad
    max_dim = max((max([cell.shape[:2] for cell in cells])))
    new_cells = [zero_pad(cell, max_dim) for cell in cells]

    # visualize
    # id = 200000
    # for i in range(7):
    #     plt.subplot(2, 7, i + 1)
    #     plt.imshow(cells[id][:, :, i], cmap='gray')
    #     plt.subplot(2, 7, i + 8)
    #     plt.imshow(new_cells[id][:, :, i], cmap='gray')
    # plt.tight_layout()
    # plt.show()

    # generate labels


if __name__ == '__main__':
    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
