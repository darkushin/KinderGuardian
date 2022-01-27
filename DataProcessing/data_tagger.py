import os
import pickle
import sys
import warnings
from collections import defaultdict
from matplotlib import pyplot as plt
from sqlalchemy import func
from DataProcessing.DB.dal import get_entries, Crop, create_session
from DataProcessing.dataProcessingConstants import *


def mark_vague(track, crop_inds):
    for crop_id in crop_inds:
        track[crop_id].is_vague = True


def relabel_all(track, new_label):
    for crop in track:
        crop.label = new_label


def discard_crops(track, crops_inds):
    for crop_id in crops_inds:
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
    new_label_id = int(input())
    if new_label_id in ID_TO_NAME.keys():
        new_label_name = ID_TO_NAME[new_label_id]
    else:
        print('Please enter a valid Id from the dict keys')
        return None
    return new_label_name


def parse_input(inp: str):
    ret = []
    arr = inp.split(' ')
    for elem in arr:
        r = elem.split('-')
        if len(r) == 1:
            ret.append(int(r[0]))  # this is a single crop_id
        elif len(r) == 2:
            ret.extend(list(range(int(r[0]), int(r[1]) + 1)))  # this is a range
        else:
            print('Invalid input, try again')
            return []
    return ret


def label_tracks_DB(vid_name: str, crops_folder: str, session):
    def _label_track_DB(session, track_query):
        """

        Args:
            track: A list of Crop objects representing the track
        Returns:
        """
        counter = 0
        actions_taken = []
        track = track_query.all()
        for batch in range(0, len(track), NUM_OF_CROPS_TO_VIEW):
            cur_batch = track[batch:min(batch + NUM_OF_CROPS_TO_VIEW, len(track))]
            _, axes = plt.subplots(X_AXIS_NUM_CROPS, Y_AXIS_NUM_CROPS, figsize=(13, 13))
            axes = axes.flatten()
            for a in axes:
                a.axis('off')
            for crop, ax in zip(cur_batch, axes):
                # using / on to adapt to windows env
                img_path = os.path.join(crops_folder, crop.im_name)
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(counter)
                counter += 1
            plt.title(f'Label == {cur_batch[0].label}')
            plt.show()

        while True:
            user_input = input(f"{APPROVE_TRACK} to approve,"
                               f"{SPLIT_TRACK} to split, "
                               f"{DISCARD} to discard,"
                               f"{VAGUE} to mark vague,"
                               f"{RELABEL} for relabel,"
                               f"{SKIP_TRACK} to skip review")

            if user_input == APPROVE_TRACK:
                print('The following actions were taken : ')
                print(actions_taken)
                if input(f"{APPROVE_TRACK} to approve, other input to cont. tagging") == APPROVE_TRACK:
                    reviewed(track)
                    break
                else:
                    continue
            if user_input == SKIP_TRACK:
                break
            elif user_input == DISCARD:  # receives a sequence of size >= 1
                # discards = [int(x) for x in input('Enter crop_ids to Discard').split()]
                discard_input = input('Enter crop_ids to Discard')
                parsed_input = parse_input(discard_input)
                discard_crops(track, parsed_input)
                if max(parsed_input) > len(track):
                    print('Some values are not valid crop ids, try again')
                    continue
                actions_taken.append((DISCARD, discard_input))

            elif user_input == VAGUE:
                # vagues = [int(x) for x in input('Enter crop_ids to set as Vague').split()]
                vague_input = input('Enter crop_ids to set as Vague')
                vagues = parse_input(vague_input)
                if max(vagues) > len(track):
                    print('Some values are not valid crop ids, try again')
                    continue
                mark_vague(track, vagues)
                actions_taken.append((VAGUE, vague_input))

            elif user_input == SPLIT_TRACK:
                start = int(input('Enter start of split'))
                end = int(input('Enter end of split')) + 1
                new_label = insert_new_label()
                if not new_label:
                    continue
                new_track_id = session.query(func.max(Crop.track_id)).scalar() + 1
                split_track(track, start, end, new_label, new_track_id)
                actions_taken.append((SPLIT_TRACK, start, end, new_label, new_track_id))

            elif user_input == RELABEL:
                new_label = insert_new_label()
                relabel_all(track, new_label)
                actions_taken.append((RELABEL, new_label))
            elif user_input == QUIT:
                save_work = input('Save changes to DB? [y/n]')
                if save_work == 'y':
                    print('Saving changes to DB and quiting')
                    session.commit()
                    sys.exit(0)
                elif save_work == 'n':
                    print('Quiting without saving changes to DB')
                    sys.exit(0)
                else:
                    print('invalid input for quiting, sit down and continue labeling!')
            else:
                warnings.warn('Please Insert one of the supported actions')

    track_ids = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}),
                                                         group=Crop.track_id, session=session)]

    for track_id in track_ids:
        print()
        try:
            track_query = get_entries(filters=(Crop.vid_name == vid_name,
                                               Crop.track_id == track_id,
                                               Crop.reviewed_one == False),
                                      order=Crop.crop_id,
                                      session=session)
            _label_track_DB(session=session, track_query=track_query)
        except Exception as e:
            warnings.warn(f'Error! {e}')
    session.commit()


if __name__ == '__main__':
    session = create_session()
    label_tracks_DB(vid_name='1.8.21-095724',
                    crops_folder="/mnt/raid1/home/bar_cohen/DB_Test/",
                    session=session)
