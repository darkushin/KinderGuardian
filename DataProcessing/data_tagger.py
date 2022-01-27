import os
import pickle
from collections import defaultdict
from matplotlib import pyplot as plt
from sqlalchemy import func
from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import ID_TO_NAME

def mark_vague(track, crop_inds):
    for crop_id in crop_inds:
        track[crop_id].is_vague = True

def relabel_all(track, new_label):
    for crop in track:
        crop.label = new_label

def discard_crops(track, crop_inds):
    for crop_id in crop_inds:
        track[crop_id].invalid = True

def split_track(track, split_start, split_end, new_label, new_track_id):
    splitted_track = track[split_start:split_end]
    for crop in splitted_track:
        crop.track_id = new_track_id
        crop.label = new_label

def reviewed(track):
    for crop in track:
        crop.reviewed = True


def insert_new_label():
    print("Please insert of of the following Ids:")
    print(ID_TO_NAME)
    new_label_name = ID_TO_NAME[int(input())]
    return new_label_name

def label_tracks_DB(vid_name:str, crops_folder:str, session):
    def _label_track_DB(session, track_query):
        """

        Args:
            track: A list of Crop objects representing the track
        Returns:
        """
        counter = 0
        NUM_OF_CROPS_TO_VIEW = 25
        track = track_query.all()
        for batch in range(0, len(track), NUM_OF_CROPS_TO_VIEW):
            cur_batch = track[batch:min(batch + NUM_OF_CROPS_TO_VIEW, len(track))]
            _, axes = plt.subplots(5, 5, figsize=(10, 10))
            axes = axes.flatten()
            for a in axes:
                a.axis('off')
            for crop, ax in zip(cur_batch, axes):
                # using / on to adapt to windows env
                img_path = os.path.join(crops_folder , crop.im_name)
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(counter)
                counter += 1
            plt.title(f'Label == {cur_batch[0].label}')
            plt.show()

        while True:
            user_input = input("Y for approve and move to next Track,"
                               "S for split, "
                               "discard for discard crops,"
                               "V for mark vague,"
                               "R for relabel,"
                               "skip if you are unsure of changes and want to skip review")

            if user_input == 'Y':
                reviewed(track)
                break
            if user_input == 'Skip':
                break
            elif user_input == 'discard':  # receives a sequence of size >= 1
                discards = [int(x) for x in input('Enter crop_ids to Discard').split()]
                discard_crops(track, discards)
            elif user_input == 'V':
                vagues = [int(x) for x in input('Enter crop_ids to set as Vague').split()]
                mark_vague(track, vagues)
            elif user_input == 'S':
                start = int(input('Enter start of split'))
                end = int(input('Enter end of split')) + 1
                new_label = insert_new_label()
                max_track_id =  session.query(func.max(Crop.track_id)).scalar() + 1
                split_track(track, start, end, new_label, max_track_id)
            elif user_input == 'R':
                new_label = insert_new_label()
                relabel_all(track, new_label)
            else:
                print('Please Insert one of the supported actions')

    track_ids = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}),
                                                         group=Crop.track_id, session=session)]

    for track_id in track_ids:
        # todo dont iter over reviewed tracks
        track_query = get_entries(filters=(Crop.vid_name == vid_name, Crop.track_id == track_id),
                                  order=Crop.crop_id,
                                  session=session)
        _label_track_DB(session=session, track_query=track_query)
        break
    session.commit()

if __name__ == '__main__':
    db_path = "/mnt/raid1/home/bar_cohen/Shoham_KG.db"
    session = create_session(db_path)
    label_tracks_DB(vid_name='1.8.21-095724' ,
                    crops_folder = "/mnt/raid1/home/bar_cohen/DB_Test/",
                    session=session)
