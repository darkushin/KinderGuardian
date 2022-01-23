import os
import pickle
from collections import defaultdict
from matplotlib import pyplot as plt


# todo all of these actions need to be done to the db, currently they all run in O(n) for each Crop changed !
from DataProcessing.dataProcessingConstants import ID_TO_NAME


def mark_vague(db, track, crop_inds):
    for crop_id in crop_inds:
        db_index = db.index(track[crop_id])
        db[db_index].is_vague = True

def relabel_all(db, track, new_label):
    for crop in track:
        db_index = db.index(crop)
        db[db_index].set_label(new_label)

def discard_crops(db, track, crop_inds):
    # todo this should be done later to keep presistance of ids, or at the end of the track
    for crop_id in crop_inds:
        db_index = db.index(track[crop_id])
        del db[db_index] # todo should we del the actual crops as well?

def split_track(db, track, split_start, split_end , new_label):
    splitted_track = track[split_start:split_end]
    new_track_id = max(crop.track_id for crop in db) + 1
    for crop in splitted_track:
        db_index = db.index(crop)
        db[db_index].track_id = new_track_id
        db[db_index].set_label(new_label)

def insert_new_label():
    print("Please insert of of the following Ids:")
    print(ID_TO_NAME)
    new_label_name = ID_TO_NAME[int(input())]
    return new_label_name


def label_tracklets(crop_db_path:str):
    def _label_tracklet(crops_pkl, track:list):
        """

        Args:
            track: A list of Crop objects representing the track
        Returns:
        """
        # todo deal with face images
        NUM_OF_CROPS_TO_VIEW = 25
        counter = 0
        for batch in range(0,len(track),NUM_OF_CROPS_TO_VIEW):
            cur_batch = track[batch:min(batch+NUM_OF_CROPS_TO_VIEW, len(track))]
            _, axes = plt.subplots(5, 5, figsize=(10,10))
            axes = axes.flatten()
            for a in axes:
                a.axis('off')
            # axes = [a.axis('off') for a in axes]
            for crop,ax in zip(cur_batch, axes):
                # using / on to adapt to windows env
                img_path = os.path.join(crop_db_path + '//' + crop.unique_crop_name + '.png')
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
            elif user_input == 'D': # recieves a sequence of size >= 1
                discards = [int(x) for x in input('Enter crop_ids to Discard').split()]
                discard_crops(crops_pkl, track, discards)
            elif user_input == 'V':
                vagues = [int(x) for x in input('Enter crop_ids to set as Vague').split()]
                mark_vague(crops_pkl, track, vagues)
            elif user_input == 'S':
                start = int(input('Enter start of split'))
                end = int(input('Enter end of split, plus 1'))
                new_label = insert_new_label()
                split_track(crops_pkl, track, start, end, new_label)
            elif user_input == 'R':
                new_label = insert_new_label()
                relabel_all(crops_pkl, track, new_label)
            else:
                print('Please Insert one of the supported actions')


    crops_pkl = pickle.load(open(os.path.join(crop_db_path, '_crop_db.pkl'),'rb')) #todo make crop_db.pkl a const
    crop_dict_by_frame = defaultdict(list)
    for crop in crops_pkl:
        crop_dict_by_frame[crop.track_id].append(crop)
    for track in crop_dict_by_frame.values():
        if len(track) > 25:
            # track = [crop for crop in track] # todo lets deal with faces later
            _label_tracklet(crops_pkl, track)

if __name__ == '__main__':
    label_tracklets('/mnt/raid1/home/bar_cohen/DB_Crops_tracktor98')

