import os
import pickle
import random
import shutil

from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import NAME_TO_ID
from DataProcessing.utils import im_name_format

BASE_CROPS_LOCATION = '/mnt/raid1/home/bar_cohen/'
# DATASET_OUTPUT_LOCATION = '/mnt/raid1/home/bar_cohen/OUR_DATASETS/DukeMTMC-VideoReID'

DATASET_OUTPUT_LOCATION = '/home/bar_cohen/KinderGuardian/fast-reid/datasets/NewData'
TRAIN_PERCENT = 0.8
QUERY_PRECENT = 0.95


def im_name_in_mars(crop: Crop, track_counter, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}C{crop.cam_id}T{track_counter:04d}F{crop_id:03d}.png'


def im_name_in_duke(crop: Crop, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}_C{crop.cam_id}_F{crop_id + 1:04d}_X{crop.frame_num:05d}.png'

def im_name_in_img_duke(crop:Crop, crop_id):
    # 0012_c2_f0211013.jpg
    return f'{NAME_TO_ID[crop.label]:04d}_c{crop.cam_id}_f{crop_id:07d}.jpg'

def convert_to_img_reid_duke_naming(dataset_path:str):
    video_names = [vid.vid_name for vid in get_entries(filters=(),group=Crop.vid_name)]
    crop_counter  = 0
    for vid_name in video_names:
        print('running', vid_name)
        set_folder = 'bounding_box_train'
        if vid_name[0:8] == '20210804':
            set_folder = 'bounding_box_test'
            if random.uniform(0, 1) > 0.75:
                set_folder = 'query'


        tracks = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}), group=Crop.track_id)]
        for track in tracks:

            track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                               Crop.track_id == track,
                                               Crop.reviewed_one == True,
                                               Crop.invalid == False)).all()

            for crop in track_crops:
                output_name = im_name_in_img_duke(crop, crop_counter)
                orig_crop_path = os.path.join(BASE_CROPS_LOCATION, crop.vid_name, crop.im_name)
                dataset_crop_path = os.path.join(dataset_path, set_folder, output_name)
                os.makedirs(os.path.join(dataset_path, set_folder), exist_ok=True)
                shutil.copy(orig_crop_path, dataset_crop_path)
                crop_counter += 1


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
            # print(f'cur track {i + 1}/{len(track_ids)}')

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
            # return
        pickle.dump(unique_track_mapping, open(os.path.join(MARS_TRACKS_LOCATION, 'track_ids_mapping.pkl'), 'wb'))


def convert_to_duke_naming(dataset_path: str):
    session = create_session()
    unique_track_mapping = {}

    # For each person iterate over each video the person appears in, get all tracks of this person from the video and
    # save them to the person's ID

    # Filter according to the different persons:
    labels = [person.label for person in get_entries(filters=(), group=Crop.label)]
    for person_id, label in enumerate(labels):
        if label in ['Guy', 'Halel']:
            continue
        print(f'{person_id}/{len(labels)} Creating tracks for: {label} ({NAME_TO_ID[label]:04d})')
        numerical_label = NAME_TO_ID[label]
        # os.makedirs(os.path.join(dataset_path, f'{numerical_label:04d}'), exist_ok=True)
        track_counter = 1

        # iterate over each video in which the current person appears
        video_names = [vid.vid_name for vid in get_entries(filters=({Crop.label == label}), group=Crop.vid_name)]
        for vid_name in video_names:
            # iterate over each track of the person in the video
            tracks = [track.track_id for track in
                      get_entries(filters=(Crop.label == label, Crop.vid_name == vid_name), group=Crop.track_id)]
            for track in tracks:
                track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                                   Crop.label == label,
                                                   Crop.track_id == track,
                                                   Crop.reviewed_one == True,
                                                   Crop.invalid == False)).all()
                is_train = random.uniform(0, 1) < TRAIN_PERCENT
                if is_train:
                    set_folder = 'train'
                else:
                    set_folder = 'gallery'

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

                track_counter += 1
        pickle.dump(unique_track_mapping, open(os.path.join(DATASET_OUTPUT_LOCATION, 'track_ids_mapping.pkl'), 'wb'))


def create_query_from_img_gallery(dataset_path, query_size = 100):
    """
    Given a gallery set with crops, move query_size of crops from the gallery to the query
    # note this is is currently random and does not depend on class dist
    """
    test_gallery = os.listdir(path=os.path.join(dataset_path, 'bounding_box_test'))
    queries = random.sample(test_gallery, query_size)
    os.makedirs(os.path.join(dataset_path, 'query'), exist_ok=True)
    for query in queries:
        shutil.move(os.path.join(dataset_path, 'bounding_box_test', query), os.path.join(dataset_path, 'query', query))

def create_query_from_video_gallery(num_tracks, dataset_path):
    """
    Given a gallery set with tracklets, move <num_tracks> tracklets from the gallery to the query
    """
    for id in os.listdir(os.path.join(dataset_path, 'gallery')):
        gallery_tracks = os.listdir(os.path.join(dataset_path, 'gallery', id))
        sample_size = num_tracks
        if len(gallery_tracks) <= num_tracks:
            if num_tracks > 1:
                sample_size = 1
            else:
                continue
        query_tracks = random.sample(gallery_tracks, sample_size)
        os.makedirs(os.path.join(dataset_path, 'query', id))
        for track in query_tracks:
            shutil.move(os.path.join(dataset_path, 'gallery', id, track), os.path.join(dataset_path, 'query', id, track))


if __name__ == '__main__':
    convert_to_img_reid_duke_naming(DATASET_OUTPUT_LOCATION)
    im_name_format(DATASET_OUTPUT_LOCATION + '/query')


