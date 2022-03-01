import os
import pickle
import random
import shutil
import warnings

import pandas as pd

from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import NAME_TO_ID
from DataProcessing.utils import im_name_format

BASE_CROPS_LOCATION = '/mnt/raid1/home/bar_cohen/'
# DATASET_OUTPUT_LOCATION = '/mnt/raid1/home/bar_cohen/OUR_DATASETS/DukeMTMC-VideoReID' # keep this, it is for video location
DATASET_OUTPUT_LOCATION = '/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0308' # this is for image


def im_name_in_mars(crop: Crop, track_counter, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}C{crop.cam_id}T{track_counter:04d}F{crop_id:03d}.png'


def im_name_in_duke(crop: Crop, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}_C{crop.cam_id}_F{crop_id + 1:04d}_X{crop.frame_num:05d}.png'

def im_name_in_img_duke(crop:Crop, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}_c{crop.cam_id}_f{crop_id:07d}.jpg'

def convert_to_img_reid_duke_naming(dataset_path:str, query_days, same_day=None):
    # entering 'same_day' will result in a split only to query, train and test only from this day

    df = pd.DataFrame(columns=['file_name', 'vid_name', 'track_id'])
    warnings.warn("Query will be taken from same day as test")
    video_names = [vid.vid_name for vid in get_entries(filters=(),group=Crop.vid_name)]
    crop_counter  = 0
    for vid_name in video_names:
        if vid_name[0:8] == same_day or vid_name in query_days:
            set_folder = 'query'
        else:
            if same_day:
                continue
            set_folder = 'bounding_box_train'
        print('running', vid_name)
        tracks = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}), group=Crop.track_id)]
        for track in tracks:
            track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                               Crop.track_id == track,
                                               Crop.reviewed_one == True,
                                               Crop.invalid == False)).all()
            if same_day and random.random() <= 0.8:
                set_folder = 'bounding_box_train'

            for crop in track_crops:
                output_name = im_name_in_img_duke(crop, crop_counter)
                orig_crop_path = os.path.join(BASE_CROPS_LOCATION, crop.vid_name, crop.im_name)
                dataset_crop_path = os.path.join(dataset_path, set_folder, output_name)
                os.makedirs(os.path.join(dataset_path, set_folder), exist_ok=True)
                shutil.copy(orig_crop_path, dataset_crop_path)
                if set_folder == 'bounding_box_train':  # add to gallery as well
                    dataset_crop_path = os.path.join(dataset_path, 'bounding_box_test', output_name)
                    os.makedirs(os.path.join(dataset_path, 'bounding_box_test'), exist_ok=True)
                    shutil.copy(orig_crop_path, dataset_crop_path)
                crop_counter += 1
                df = add_file_data_base(df, output_name, crop)
    df.to_csv(os.path.join(DATASET_OUTPUT_LOCATION, 'img_to_info.csv'))


def convert_to_mars_naming(dataset_path: str):
    session = create_session()
    unique_track_mapping = {}

    # For each person iterate over each video the person appears in, get all tracks of this person from the video and
    # save them to the person's ID
    # Filter according to the different persons:
    labels = [person.label for person in get_entries(filters=(), group=Crop.label)]
    for person_id, label in enumerate(labels):
        print(f'{person_id}/{len(labels)} Creating tracks for: {label} ({NAME_TO_ID[label]:04d})')
        numerical_label = NAME_TO_ID[label]
        os.makedirs(os.path.join(dataset_path, f'{numerical_label:04d}'), exist_ok=True)
        track_counter = 1

        # iterate over each video in which the current person appears
        video_names = [vid.vid_name for vid in get_entries(filters=({Crop.label == label}), group=Crop.vid_name)]
        for vid_name in video_names:
            # iterate over each track of the person in the video
            tracks = [track.track_id for track in get_entries(filters=(Crop.label == label, Crop.vid_name == vid_name), group=Crop.track_id)]
            for track in tracks:
                track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                                   Crop.label == label,
                                                   Crop.track_id == track,
                                                   Crop.reviewed_one == True,
                                                   Crop.invalid == False)).all()

                if not track_crops:
                    continue
                for crop_id, crop in enumerate(track_crops):
                    unique_track_mapping[f'{vid_name}_{label}_{crop.track_id}'] = track_counter
                    output_name = im_name_in_mars(crop, track_counter, crop_id)
                    orig_crop_path = os.path.join(BASE_CROPS_LOCATION, crop.vid_name, crop.im_name)
                    mars_crop_path = os.path.join(dataset_path, f'{numerical_label:04d}', output_name)
                    shutil.copy(orig_crop_path, mars_crop_path)

                track_counter += 1
        pickle.dump(unique_track_mapping, open(os.path.join('MARS_TRACKS_LOCATION', 'track_ids_mapping.pkl'), 'wb'))

def add_file_data_base(df:pd.DataFrame, file_name:str, crop:Crop):
    df = df.append({'file_name':file_name, 'vid_name':crop.vid_name, 'track_id':crop.track_id}, ignore_index=True)
    return df

def convert_to_video_reid_duke_naming(dataset_path: str, query_day):
    df = pd.DataFrame(columns=['file_name', 'vid_name', 'track_id'])
    session = create_session()
    unique_track_mapping = {}

    # For each person iterate over each video the person appears in, get all tracks of this person from the video and
    # save them to the person's ID

    # Filter according to the different persons:
    labels = [person.label for person in get_entries(filters=(), group=Crop.label)]
    for person_id, label in enumerate(labels):
        print(f'{person_id}/{len(labels)} Creating tracks for: {label} ({NAME_TO_ID[label]:04d})')
        numerical_label = NAME_TO_ID[label]
        track_counter = 1

        # iterate over each video in which the current person appears
        video_names = [vid.vid_name for vid in get_entries(filters=({Crop.label == label}), group=Crop.vid_name)]
        for vid_name in video_names:
            # iterate over each track of the person in the video
            tracks = [track.track_id for track in
                      get_entries(filters=(Crop.label == label, Crop.vid_name == vid_name), group=Crop.track_id)]
            for track in tracks:
                # if vid_name[0:8] == test_day:
                #     set_folder = 'gallery'
                if vid_name[0:8] in query_day:
                    set_folder = 'query'
                else:
                    set_folder = 'train'
                track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                                   Crop.label == label,
                                                   Crop.track_id == track,
                                                   Crop.reviewed_one == True,
                                                   Crop.invalid == False)).all()
                if not track_crops:
                    continue
                for crop_id, crop in enumerate(track_crops):
                    unique_track_mapping[f'{label}_{vid_name}_{crop.track_id}'] = track_counter
                    output_name = im_name_in_duke(crop, crop_id)
                    orig_crop_path = os.path.join(BASE_CROPS_LOCATION, crop.vid_name, crop.im_name)
                    dataset_crop_path = os.path.join(dataset_path, set_folder, f'{numerical_label:04d}',
                                                     f'{track_counter:04d}', output_name)
                    os.makedirs(
                        os.path.join(dataset_path, set_folder, f'{numerical_label:04d}', f'{track_counter:04d}'),
                        exist_ok=True)

                    shutil.copy(orig_crop_path, dataset_crop_path)
                    if set_folder == 'train': # add to gallery as well
                        dataset_crop_path = os.path.join(dataset_path, 'gallery', f'{numerical_label:04d}',
                                                         f'{track_counter:04d}', output_name)
                        os.makedirs(
                            os.path.join(dataset_path, 'gallery', f'{numerical_label:04d}', f'{track_counter:04d}'),
                            exist_ok=True)
                        shutil.copy(orig_crop_path, dataset_crop_path)

                    df = add_file_data_base(df,output_name, crop)
                track_counter += 1
    df.to_csv(os.path.join(dataset_path, 'img_to_info.csv'))
    pickle.dump(unique_track_mapping, open(os.path.join(dataset_path, 'track_ids_mapping.pkl'), 'wb'))


def create_query_from_gallery(dataset_path, is_video=False):
    """
    Given a gallery set with crops, move query_size of crops from the gallery to the query
    # note this is is currently random and does not depend on class dist
    """
    GALLERY_NAME = 'bounding_box_test' if not is_video else 'gallery'
    test_gallery = os.listdir(path=os.path.join(dataset_path, GALLERY_NAME))
    test_gallery_ids = {im[0:4] for im in test_gallery}
    queries = os.listdir(path=os.path.join(dataset_path, 'query'))

    # if the query does not exists in test gallery, move it there
    for query in queries:
        if query[0:4] not in test_gallery_ids:
            shutil.move(os.path.join(dataset_path, 'query', query), os.path.join(dataset_path, GALLERY_NAME, query))

if __name__ == '__main__':
    # convert_to_duke_naming(DATASET_OUTPUT_LOCATION, test_day='20210804',query_day='20210803')
    convert_to_img_reid_duke_naming(DATASET_OUTPUT_LOCATION, query_days=['3007, 0808'])
    create_query_from_gallery(DATASET_OUTPUT_LOCATION, is_video=False)
    im_name_format(DATASET_OUTPUT_LOCATION + '/query', is_video=False)



