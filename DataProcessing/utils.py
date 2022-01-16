import os
import pickle
import tempfile
from collections import defaultdict

import cv2
import numpy as np
from mmtrack.core.utils.visualization import _cv2_show_tracks as plot_tracks
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


def viz_data_on_video(input_vid, output_path, pre_labeled_pkl_path=None,path_to_crops=None):
    """
    This func assumes that the input video has been run by the track and reid model data creator to
    create a pre-annoted set.
    Args:
        input_vid:
        pre_labeled_pkl_path:

    Returns:

    """
    assert pre_labeled_pkl_path or path_to_crops , "You must enter either a pkl to cropsDB or the crops folder"
    crops = None
    if path_to_crops:
        assert os.path.isdir(path_to_crops) , "Path must be a CropDB folder"
        crops = []
        for file in os.listdir(path_to_crops):
            crop_path = os.path.join(path_to_crops, file)
            # crops.append(create_Crop_from_str(crop_path)) # todo this exists in FaceDetect branch
    elif pre_labeled_pkl_path:
        assert os.path.isfile(pre_labeled_pkl_path) , "Path must be a CropDB file"
        crops = pickle.load(open(pre_labeled_pkl_path, 'rb'))

    # create frame_to_crops dict
    crop_dict_by_frame = defaultdict(list)
    for crop in crops:
        crop_dict_by_frame[crop.frame_id].append(crop)

    imgs = mmcv.VideoReader(input_vid)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    fps = int(imgs.fps)
    for i,frame in enumerate(imgs):
        cur_crops = crop_dict_by_frame[i]
        crops_bboxes = [np.append(crop.bbox, [1]) for crop in cur_crops] ## adding 1 for keeping up with plot requirements
        crops_labels = [crop.label for crop in cur_crops]
        cur_img = plot_tracks(img=frame,bboxes=np.array(crops_bboxes), ids=np.array(crops_labels), labels=np.array(crops_labels))
        mmcv.imwrite(cur_img, f'{temp_path}/{i:03d}.png')
    mmcv.frames2video(temp_path, output_path, fps=fps, fourcc='mp4v', filename_tmpl='{:03d}.png')
    temp_dir.cleanup()

def trim_video(input_path, output_path, limit):
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
    viz_data_on_video(input_vid="/home/bar_cohen/KinderGuardian/Videos/trimmed_1.8.21-095724.mp4",
                      output_path="/home/bar_cohen/KinderGuardian/Results/trimmed_1.8.21-095724_labeld.mp4",
                      pre_labeled_pkl_path="/mnt/raid1/home/bar_cohen/DB_Crops/_crop_db.pkl")
    # rename_folders = ['third-query-2.8_test-4.8/bounding_box_train']
    # for folder in rename_folders:
    #     remove_images_from_dataset(f'/home/bar_cohen/KinderGuardian/fast-reid/datasets/{folder}', 'f03')

