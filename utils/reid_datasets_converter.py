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

DATASET_PATH = '/home/bar_cohen/raid/42street/CCVID-Datasets/'
DUKE_DATASET_PATH = '/home/bar_cohen/raid/42street/fast_reid_tests/query_part1_s9500_e10001'
STREET_CROPS_PATH = '/home/bar_cohen/raid/42street/42StreetCrops'

def create_dataset_dir_tree(dataset_path):
    """
    Create all empty directories under the dataset_path according to the structure of the PRCC dataset.
    """
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/A'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/B'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/test/C'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'prcc/rgb/val'), exist_ok=True)


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

        # iterate over all non-invalid images in the video:
        images = [(image.im_name, image.label) for image in get_entries(filters=(Crop.vid_name == video, Crop.invalid == False), db_path=db_location).all()]

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


def get_images_from_DB(video_name: str, db_location: str = DB_LOCATION):
    """ Given a video that was reviewed and exists in the DB, retrieve all images from this video that are not invalid"""
    reviewed_videos = [vid.vid_name for vid in get_entries(filters=({Crop.reviewed_one == True}), group=Crop.vid_name).all()]
    assert video_name in reviewed_videos, f'Video {video_name} was not reviewed yet, labels might be wrong. Double check if this is intended.'
    vid_images = [(image.im_name, image.label) for image in get_entries(filters=(Crop.vid_name == video_name, Crop.invalid == False), db_path=db_location).all()]
    return vid_images


def add_gallery_images(gallery_paths, output_path):
    """
    Copy all images from every gallery in the gallery_paths list to the output_path.
    Assumes each gallery under `gallery_paths` has a bounding_box_test folder with all images and there labels in the XXXX_cY_fZZZZZZZ.jpg format.
    Copies all images to the output path under folders for each id to match the PRCC format.
    """
    for gallery in gallery_paths:
        gallery_path = os.path.join(gallery, 'bounding_box_test')
        for im_name in os.listdir(gallery_path):
            im_id = int(im_name.split('_')[0])
            im_dir = os.path.join(output_path, f'{im_id:03d}')
            os.makedirs(im_dir, exist_ok=True)
            shutil.copyfile(src=os.path.join(gallery_path, im_name), dst=os.path.join(im_dir, im_name))


def create_readme_file(query_videos, gallery_paths, new_dataset_path, query_videos_C):
    """
    Creates a README file in the dataset folder which explains the configurations of this dataset
    """
    readme = open(os.path.join(new_dataset_path, "README.md"), "w+")
    readme.write(f'The dataset contains the following images: \n'
                 f'- **A (gallery) **: {gallery_paths}\n'
                 f'- **B (query) **: {query_videos}\n')
    if query_videos_C:
        readme.write(f'- **C (additional query) **: {query_videos_C}\n')
    readme.close()


def create_PRCC_dataset(query_videos: list, gallery_paths: list, dataset_name: str, query_videos_C: list = None):
    """
    This function assumes the query videos should be taken from the DB and the gallery is a gallery created by our
    unsupervised flow. The gallery should hold all images under gallery_path/bounding_box_test
    """
    # Create all directories for the dataset:
    new_dataset_path = os.path.join(DATASET_PATH, dataset_name)
    create_dataset_dir_tree(new_dataset_path)

    # Add all query images to the B folder in the dataset
    for video in query_videos:
        vid_images = get_images_from_DB(video)
        copy_images(origin_images=vid_images, dest=os.path.join(new_dataset_path, 'prcc/rgb/test/B'),
                    crops_location=os.path.join(STREET_CROPS_PATH, video))

    # If given query videos for C folder, add them to the C folder
    if query_videos_C:
        for video in query_videos_C:
            vid_images = get_images_from_DB(video)
            copy_images(origin_images=vid_images, dest=os.path.join(new_dataset_path, 'prcc/rgb/test/C'),
                        crops_location=os.path.join(STREET_CROPS_PATH, video))

    add_gallery_images(gallery_paths, os.path.join(new_dataset_path, 'prcc/rgb/test/A'))

    create_readme_file(query_videos, gallery_paths, new_dataset_path, query_videos_C)


create_PRCC_dataset(query_videos=['part3_s22000_e22501'], gallery_paths=['/home/bar_cohen/raid/42street/part3_all'],
                    dataset_name="query_part3_gallery_part3", query_videos_C=['part3_s28500_e29001'])

# os.makedirs(DUKE_DATASET_PATH, exist_ok=True)
# create_dataset(['part1'], ['part2'], '/home/bar_cohen/raid/42street/42StreetCrops')
# create_duke_dataset(['part2'], ['part1'], '/home/bar_cohen/raid/42street/42StreetCrops')

