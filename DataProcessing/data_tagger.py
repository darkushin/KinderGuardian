import os
import pickle
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

# todo all of these actions need to be done to the db, currently they all run in O(n) for each Crop changed !
from DataProcessing.DB.dal import get_entries, Crop, generate_new_track_id, create_session
from DataProcessing.dataProcessingConstants import ID_TO_NAME

#
# def mark_vague(pkl, track, crop_inds):
#     for crop_id in crop_inds:
#         pkl_index = pkl.index(track[crop_id])
#         pkl[pkl_index].is_vague = True
#
#
# def relabel_all(pkl, track, new_label):
#     for crop in track:
#         pkl_index = pkl.index(crop)
#         pkl[pkl_index].set_label(new_label)
#
#
# def discard_crops(pkl, track, crop_inds):
#     # todo this should be done later to keep presistance of ids, or at the end of the track
#     for crop_id in crop_inds:
#         pkl_index = pkl.index(track[crop_id])
#         del pkl[pkl_index]  # todo should we del the actual crops as well?
#
#
# def split_track(pkl, track, split_start, split_end, new_label):
#     splitted_track = track[split_start:split_end]
#     new_track_id = max(crop.track_id for crop in pkl) + 1
#     for crop in splitted_track:
#         pkl_index = pkl.index(crop)
#         pkl[pkl_index].track_id = new_track_id
#         pkl[pkl_index].set_label(new_label)
#
#
# def insert_new_label():
#     print("Please insert of of the following Ids:")
#     print(ID_TO_NAME)
#     new_label_name = ID_TO_NAME[int(input())]
#     return new_label_name


def mark_vague(track, crop_inds):
    for crop_id in crop_inds:
        track[crop_id].is_vague = True




def relabel_all(session, track, new_label):
    for crop in track:
        # crop.update({Crop.label : new_label})
        crop.label = new_label
        session.commit()

def discard_crops(track, crop_inds):
    # todo do we really want to do this?
    for crop_id in crop_inds:
        track[crop_id].delete()


def split_track(track, split_start, split_end, new_label, new_track_id):
    splitted_track = track[split_start:split_end]

    for crop in splitted_track:
        crop.track_id = new_track_id
        crop.label = new_label


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
        NUM_OF_CROPS_TO_VIEW = 25
        counter = 0
        track = track_query.all()
        relabel_all(session , track,'maol')
        return
        for batch in range(0, len(track), NUM_OF_CROPS_TO_VIEW):
            cur_batch = track[batch:min(batch + NUM_OF_CROPS_TO_VIEW, len(track))]
            _, axes = plt.subplots(5, 5, figsize=(10, 10))
            axes = axes.flatten()
            for a in axes:
                a.axis('off')
            # axes = [a.axis('off') for a in axes]
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
            user_input = input("Y for next Track, S for split, D for discard crops, V for mark vauge, R for relabel")
            if user_input == 'Y':
                break
            elif user_input == 'D':  # receives a sequence of size >= 1
                discards = [int(x) for x in input('Enter crop_ids to Discard').split()]
                discard_crops(track_query, discards)
            elif user_input == 'V':
                vagues = [int(x) for x in input('Enter crop_ids to set as Vague').split()]
                mark_vague(track_query, vagues)
            elif user_input == 'S':
                start = int(input('Enter start of split'))
                end = int(input('Enter end of split, plus 1'))
                new_label = insert_new_label()
                new_track_id = generate_new_track_id(session)
                split_track(track_query, start, end, new_label, new_track_id)
            elif user_input == 'R':
                new_label = insert_new_label()
                relabel_all(track_query, new_label)
            elif user_input == 'STOP IT':
                return
            else:
                print('Please Insert one of the supported actions')

    # crops_pkl = pickle.load(open(os.path.join(crop_pkl_path, '_crop_db.pkl'), 'rb'))  # todo make crop_db.pkl a const

    # get all unique track ids from DB
    # for each track id get all crops of track
    # run submodel of relabel track

    track_ids = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}),
                                                         group=Crop.track_id, session=session)]
    # tracks = np.unique([crop.track_id for crop in video_crops])
    for track_id in track_ids:
        track_query = get_entries(filters=(Crop.vid_name == vid_name, Crop.track_id == track_id),
                                  order=Crop.crop_id,
                                  session=session)
        _label_track_DB(session=session, track_query=track_query)

    #
    # crop_dict_by_frame = defaultdict(list)
    # for crop in crops_pkl:
    #     crop_dict_by_frame[crop.track_id].append(crop)
    # for track in crop_dict_by_frame.values():
    #     if len(track) > 25:
    #         # track = [crop for crop in track] # todo lets deal with faces later
    #         _label_tracklet(crops_pkl, track)

# def label_tracklets(crop_pkl_path: str):
#     def _label_tracklet(crops_pkl, track: list):
#         """
#
#         Args:
#             track: A list of Crop objects representing the track
#         Returns:
#         """
#         # todo deal with face images
#         NUM_OF_CROPS_TO_VIEW = 25
#         counter = 0
#         for batch in range(0, len(track), NUM_OF_CROPS_TO_VIEW):
#             cur_batch = track[batch:min(batch + NUM_OF_CROPS_TO_VIEW, len(track))]
#             _, axes = plt.subplots(5, 5, figsize=(10, 10))
#             axes = axes.flatten()
#             for a in axes:
#                 a.axis('off')
#             # axes = [a.axis('off') for a in axes]
#             for crop, ax in zip(cur_batch, axes):
#                 # using / on to adapt to windows env
#                 img_path = os.path.join(crop_pkl_path + '//' + crop.unique_crop_name + '.png')
#                 img = plt.imread(img_path)
#                 ax.imshow(img)
#                 ax.set_title(counter)
#                 counter += 1
#             plt.title(f'Label == {cur_batch[0].label}')
#             plt.show()
#         while True:
#             user_input = input("Y for next Track, S for split, D for discard crops, V for mark vauge, R for relabel")
#             if user_input == 'Y':
#                 break
#             elif user_input == 'D':  # receives a sequence of size >= 1
#                 discards = [int(x) for x in input('Enter crop_ids to Discard').split()]
#                 discard_crops(crops_pkl, track, discards)
#             elif user_input == 'V':
#                 vagues = [int(x) for x in input('Enter crop_ids to set as Vague').split()]
#                 mark_vague(crops_pkl, track, vagues)
#             elif user_input == 'S':
#                 start = int(input('Enter start of split'))
#                 end = int(input('Enter end of split, plus 1'))
#                 new_label = insert_new_label()
#                 split_track(crops_pkl, track, start, end, new_label)
#             elif user_input == 'R':
#                 new_label = insert_new_label()
#                 relabel_all(crops_pkl, track, new_label)
#             else:
#                 print('Please Insert one of the supported actions')
#
#     crops_pkl = pickle.load(open(os.path.join(crop_pkl_path, '_crop_db.pkl'), 'rb'))  # todo make crop_db.pkl a const
#     crop_dict_by_frame = defaultdict(list)
#     for crop in crops_pkl:
#         crop_dict_by_frame[crop.track_id].append(crop)
#     for track in crop_dict_by_frame.values():
#         if len(track) > 25:
#             # track = [crop for crop in track] # todo lets deal with faces later
#             _label_tracklet(crops_pkl, track)


if __name__ == '__main__':
    db_path = "/mnt/raid1/home/bar_cohen/Shoham_KG.db"
    session = create_session(db_path)
    label_tracks_DB(vid_name='1.8.21-095724' ,
                    crops_folder = "/mnt/raid1/home/bar_cohen/DB_Test/",
                    session=session)
