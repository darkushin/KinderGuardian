import os
import pickle
import shutil

from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import NAME_TO_ID

BASE_CROPS_LOCATION = '/mnt/raid1/home/bar_cohen/'
MARS_TRACKS_LOCATION = '/mnt/raid1/home/bar_cohen/MARS_TRACKS2'


def im_name_in_mars(crop: Crop, track_counter, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}C{crop.cam_id}T{track_counter:04d}F{crop_id:03d}.png'


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


if __name__ == '__main__':
    convert_to_mars_naming(MARS_TRACKS_LOCATION)