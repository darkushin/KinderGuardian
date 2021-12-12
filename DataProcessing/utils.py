import os
from cv2 import imread
from pathlib import Path
from tqdm import tqdm


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


def read_labled_croped_images(file_dir) -> {}:
    """
    used to load images from a folder, recursively.
    file-dir: dir to load images from
    return an ID to images dict.
    """

    # '/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned' cur path for labled data

    assert os.path.isdir(file_dir) , 'Read labled data must get a valid dir path'
    imgs = {}
    for img in tqdm(Path(file_dir).rglob('*.jpg')):
        img = str(img)
        if not os.path.isfile(img):
            continue
        img_left_bracket = img.rfind('/')
        img_id = img[img_left_bracket+1:img_left_bracket+5] # xxxx id format
        imgs[img_id] = imread(img)
    return imgs


if __name__ == '__main__':
    read_labled_croped_images('/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned')
    # rename_folders = ['third-query-2.8_test-4.8/bounding_box_train']
    # for folder in rename_folders:
    #     remove_images_from_dataset(f'/home/bar_cohen/KinderGuardian/fast-reid/datasets/{folder}', 'f03')
    #
