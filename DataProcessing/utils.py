import os

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
        new_im_name = im.replace('c1', 'c2')

        os.rename(f'{path}/{im}', f'{path}/{new_im_name}')


if __name__ == '__main__':
    rename_dates = ['30.7.21_cam2']
    for date in rename_dates:
        im_name_format(f'/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned/{date}/')