import os
import pickle
import shutil

from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import NAME_TO_ID

BASE_CROPS_LOCATION = '/mnt/raid1/home/bar_cohen/'
MARS_TRACKS_LOCATION = '/mnt/raid1/home/bar_cohen/MARS_TRACKS'

def im_name_in_mars(crop: Crop, crop_id):
    return f'{NAME_TO_ID[crop.label]:04d}C{crop.cam_id}T{crop.track_id:04d}F{crop_id:03d}.png'

def convert_to_mars_naming(dataset_path: str):
    session = create_session()
    unique_track_mapping = {}
    track_counter = 0
    vid_names = [vid.vid_name for vid in get_entries(filters=(),group=Crop.vid_name, session=session)]
    for vid_id, vid_name in enumerate(vid_names):
        print(f'{vid_id}/{len(vid_names)} Creating for video {vid_name}')

        video_labels = [crop.label for crop in get_entries(filters=({Crop.vid_name == vid_name}),
                                                             group=Crop.label)]

        for i, label in enumerate(video_labels):
            # print(f'cur track {i + 1}/{len(track_ids)}')
            track_crops = get_entries(filters=(Crop.vid_name == vid_name,
                                               Crop.label == label,
                                               Crop.reviewed_one == True,
                                               Crop.invalid == False,
                                               ),
                                      order=Crop.track_id,
                                      session=session).all()

            if not track_crops:
                continue
            os.makedirs(os.path.join(dataset_path, f'{NAME_TO_ID[label]:04d}'), exist_ok=True)
            for crop_id, crop in enumerate(track_crops):
                unique_track_mapping[f'{vid_name}_{label}_{crop.track_id}'] = track_counter
                output_name = im_name_in_mars(crop, crop_id)
                orig_crop_path = os.path.join(BASE_CROPS_LOCATION, crop.vid_name, crop.im_name)
                mars_crop_path = os.path.join(dataset_path, str(track_counter), output_name)
                shutil.copy(orig_crop_path, mars_crop_path)

            track_counter += 1
            return
        pickle.dump(unique_track_mapping, open(os.path.join(MARS_TRACKS_LOCATION, 'track_ids_mapping.pkl'), 'wb'))


if __name__ == '__main__':
    convert_to_mars_naming(MARS_TRACKS_LOCATION)