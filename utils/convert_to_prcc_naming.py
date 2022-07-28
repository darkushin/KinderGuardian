"""
This script is used to convert the data from the DB to the naming and folder convention of different datasets.
- PRCC dataset (https://www.isee-ai.cn/~yangqize/clothing.html).

PRCC dataset format:
- prcc
  - rgb
    - test
      - A                           # holds the images that are used for the gallery
        - 001
          - cropped_rgbXXX.png
        - 002
      - B                           # images that will appear as same-clothes as the gallery
      - C                           # images that will appear as different-clothes from the gallery
    - val
      - 001
        - A_cropped_rgbXXX.png
        - C_cropped_rgbXXX.png
      - 002
    - train
      - 001
        - C_cropped_rgbXXX.png
        - B_cropped_rgbXXX.png
      - 002

DukeMTMC dataset format:
- <dataset_name>
  - bounding_box_test
    - XXXX_cY_fZZZZZZZ.jpg
  - bounding_box_train
  - query

This script creates only the test part of the dataset as we don't aim to the train the model with our data.

dirs=( 001 002 003 005 007 010 013 014 009 006 008 011 004 )
for dir in "${dirs[@]}"; do mkdir ${dir};  done
for dir in "${dirs[@]}"; do mv 0${dir}* ${dir};  done

"""

import os
from DataProcessing.DB.dal import *
from DataProcessing.dataProcessingConstants import NAME_TO_ID
import shutil

DATASET_PATH = '/home/bar_cohen/raid/42street/CCVID-Datasets/our-data-PRCC-format'
DUKE_DATASET_PATH = '/home/bar_cohen/raid/42street/fast_reid_tests/query_part1_s9500_e10001'


def create_dataset_dir_tree(dataset_path):
    """
    Create all empty directories under the dataset_path according to the structure of the PRCC dataset.
    """
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/A'))
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/B'))
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/C'))
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/train'))
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/val'))


def copy_images(origin_images, dest, crops_location):
    """
    Copy the images from crops_location to the new destination.
    """
    for image_path, label in origin_images:

        label_id = NAME_TO_ID.get(label)
        id_folder = os.path.join(dest, f'{label_id:03d}')
        os.makedirs(id_folder, exist_ok=True)
        jpg_image = f'{image_path.split(".png")[0]}.jpg'
        shutil.copyfile(src=os.path.join(crops_location, image_path), dst=os.path.join(id_folder, jpg_image))


def duke_copy_images(origin_images, dest, crops_location):
    """
    Copy the images from crops_location to the new destination.
    """
    os.makedirs(dest, exist_ok=True)
    im_idx = 1
    for image_path, label in origin_images:
        camera = '2' if 'query' in dest else '1'
        label_id = NAME_TO_ID.get(label)
        jpg_image = f'{label_id:04d}_c{camera}_f{im_idx:07d}.jpg'
        shutil.copyfile(src=os.path.join(crops_location, image_path), dst=os.path.join(dest, jpg_image))
        im_idx += 1


def is_gallery_video(gallery_part, video_name):
    """
    Given a list with the parts that should be used for the gallery and a video name, return true if the video is part
    of the gallery and false otherwise.
    """
    vid_part = video_name.split('_')[0]
    assert 'part' in vid_part, 'Video name does not match the naming convention (`partX_sYYYYY_eZZZZZ`)'
    if vid_part in gallery_part:
        return True
    return False


def is_query_video(query_part, video_name):
    """
    Given a list with the parts that should be used for the query and a video name, return true if the video is part
    of the query and false otherwise.
    """
    if 'part1_s9500_e10001' not in video_name:
        return False
    vid_part = video_name.split('_')[0]
    assert 'part' in vid_part, 'Video name does not match the naming convention (`partX_sYYYYY_eZZZZZ`)'
    if vid_part in query_part:
        print(f'Using video {video_name}')
        return True
    return False


def create_dataset(gallery_part: list, query_part: list, crops_location: str, dataset_path: str = DATASET_PATH,
                   db_location: str = DB_LOCATION):
    """
    gallery_part: the parts of the play that should be used for gallery images. List with all parts, e.g. [part1, part2]
    query_part: the parts of the play that should be used for query images. List with all parts, e.g. [part3, part4]
    crops_location: the location of the image crops in the DB from which the images can be copied to the new dataset
    dataset_path: path of the base folder in which the dataset should be created. Assumes the dir tree already exists
    """
    # iterate over all reviewed videos:
    videos = [vid.vid_name for vid in get_entries(filters=({Crop.reviewed_one == True}), group=Crop.vid_name).all()]
    for video in videos:

        # iterate over all not invalid images in the video:
        images = [(image.im_name, image.label) for image in get_entries(filters=(Crop.vid_name == video, Crop.invalid == False)).all()]

        # copy the images to the corresponding gallery/query folder according to the part of the video
        if is_gallery_video(gallery_part, video):
            copy_images(origin_images=images, dest=os.path.join(dataset_path, 'prcc/rgb/test/A'),
                        crops_location=os.path.join(crops_location, video))
        elif is_query_video(query_part, video):
            copy_images(origin_images=images, dest=os.path.join(dataset_path, 'prcc/rgb/test/B'),
                        crops_location=os.path.join(crops_location, video))
        else:
            print(f'Warning: part {video.split("_")[0]} was not set as gallery/query part. Ignoring images of video '
                  f'{video}')


def create_duke_dataset(gallery_part: list, query_part: list, crops_location: str, dataset_path: str = DATASET_PATH,
                        db_location: str = DB_LOCATION):
    """
    gallery_part: the parts of the play that should be used for gallery images. List with all parts, e.g. [part1, part2]
    query_part: the parts of the play that should be used for query images. List with all parts, e.g. [part3, part4]
    crops_location: the location of the image crops in the DB from which the images can be copied to the new dataset
    dataset_path: path of the base folder in which the dataset should be created. Assumes the dir tree already exists
    """
    # iterate over all reviewed videos:
    videos = [vid.vid_name for vid in get_entries(filters=({Crop.reviewed_one == True}), group=Crop.vid_name).all()]
    for video in videos:

        # iterate over all not invalid images in the video:
        images = [(image.im_name, image.label) for image in get_entries(filters=(Crop.vid_name == video, Crop.invalid == False)).all()]

        # copy the images to the corresponding gallery/query folder according to the part of the video
        if is_gallery_video(gallery_part, video):
            duke_copy_images(origin_images=images, dest=os.path.join(DUKE_DATASET_PATH, 'bounding_box_test'),
                        crops_location=os.path.join(crops_location, video))
            copy_images(origin_images=images, dest=os.path.join(DUKE_DATASET_PATH, 'bounding_box_train'),   # todo: change this
                        crops_location=os.path.join(crops_location, video))
        elif is_query_video(query_part, video):
            duke_copy_images(origin_images=images, dest=os.path.join(DUKE_DATASET_PATH, 'query'),
                        crops_location=os.path.join(crops_location, video))
        else:
            print(f'Warning: part {video.split("_")[0]} was not set as gallery/query part. Ignoring images of video '
                  f'{video}')


os.makedirs(DUKE_DATASET_PATH, exist_ok=True)
# create_dataset(['part1'], ['part2'], '/home/bar_cohen/raid/42street/42StreetCrops')
create_duke_dataset(['part2'], ['part1'], '/home/bar_cohen/raid/42street/42StreetCrops')

