import os
from collections import defaultdict

from cv2 import imread
from pathlib import Path
from tqdm import tqdm

import tempfile

import mmcv

"""
This folder holds functions that can be useful for data handling, such as renaming images etc.
"""


def im_name_format(path):
    """
    Convert all images in the given path from its current malformed format to the `xxxx_c1_f1234567.jpg` format which is
     the correct format for the DukeMTMC dataset.
     Change this function according to the current corrections you need to do.
    """
    for im in os.listdir(os.path.join(path)):
        if '.jpg' not in im:
            continue
        # new_im_name = im.split('.jpg')[0]
        new_im_name = im.replace('c1', 'c6')

        os.rename(f'{path}/{im}', f'{path}/{new_im_name}')


def im_id_format(path):
    """
    Reformat all images in the given directory so that the id of the person is different according to the different days.
    """
    for im in os.listdir(os.path.join(path)):
        day_num = im.split('_')[-1][1:3]
        new_id_name = day_num + im[2:]

        os.rename(f'{path}/{im}', f'{path}/{new_id_name}')


def remove_images_from_dataset(path, pattern):
    """
    Remove images that are matching the given pattern from a dataset folder.
    """
    for im in os.listdir(path):
        if pattern in im:
            os.remove(os.path.join(path, im))


def read_labeled_croped_images(file_dir, file_type='jpg') -> dict:
    """
    used to load images from a folder, recursively and returns dict with mapping between
    each image crop and it's id
    file-dir: dir to load images from
    return an ID to images dict.
    """

    # '/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned' cur path for labled data

    assert os.path.isdir(file_dir) , 'Read labeled data must get a valid dir path'
    imgs = defaultdict(list)
    print('reading imgs from dir...')
    for img in tqdm(Path(file_dir).rglob(f'*.{file_type}')):
        img = str(img)
        if not os.path.isfile(img):
            continue
        im_path = os.path.split(img)[1]
        img_id = im_path[:4] # xxxx id format
        imgs[img_id].append(imread(img))
    return imgs


def trim_video(input_path, output_path, limit):
    """
    Given a path to a video and the number of frames that should be taken from it, trim the video to the first `limit` frames. Saves the output to the `output_path` location.
    """
    imgs = mmcv.VideoReader(input_path)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    fps = int(imgs.fps)
    print('Starting to save imgs:')
    for i, img in enumerate(imgs):
        if not i % 100:
            print(f'{i} frames done.')
        if i > limit:
            break
        mmcv.imwrite(img, f'{temp_path}/{i:03d}.png')
    mmcv.frames2video(temp_path, output_path, fps=fps, fourcc='mp4v', filename_tmpl='{:03d}.png')
    temp_dir.cleanup()


if __name__ == '__main__':
    trim_video('/home/bar_cohen/Data-Shoham/1.8.21_cam1/videos/IPCamera_20210801095724.avi',
               '/home/bar_cohen/KinderGuardian/Videos/trimmed_1.8.21-095724.mp4', limit=500)

