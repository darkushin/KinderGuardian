import glob
import os
from cv2 import imread
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
import tempfile
from collections import defaultdict
import numpy as np
from mmtrack.core.utils.visualization import random_color
import mmcv
from DataProcessing.DB.dal import *
from matplotlib import  pyplot as plt
import pandas as pd
import seaborn as sns

"""
This folder holds functions that can be useful for data handling, such as renaming images etc.
"""


COLOR_TO_RGB = {
    'yellow': [1, 1, 0],
    'blue': [0, 0, 1],
    'green': [0, 1, 0],
    'red': [1, 0, 0]
}

def resize_images(input_path, output_path, size):
    """
    Resize all images in the input_path according to the given size and place them in the output_path
    """
    os.makedirs(output_path, exist_ok=True)
    for im_path in os.listdir(input_path):
        original_image = cv2.imread(os.path.join(input_path, im_path))
        im = cv2.resize(original_image, size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_path, im_path), im)


def create_video_from_imgs(input_path, output_path, fps=24):
    """
    Given a folder of images, create a video from these images.
    """
    mmcv.frames2video(input_path, output_path, fps=fps, fourcc='mp4v', filename_tmpl='{:05d}.png')


def im_name_format(path, is_video=False):
    """
    Convert all images in the given path from its current malformed format to the `xxxx_c1_f1234567.jpg` format which is
     the correct format for the DukeMTMC dataset.
     Change this function according to the current corrections you need to do.
    """
    C1 = 'c1'
    C6 = 'c6'
    if is_video:
        C1 = 'C1'
        C6 = 'C6'
    for p, subdirs, files in os.walk(path):
        for im in files:
            if '.png' not in im and '.jpg' not in im:
                continue
            # new_im_name = im.split('.jpg')[0]
            new_im_name = im.replace(C1, C6)
            # selected.add(new_im_name[0:4])
            os.rename(f'{p}/{im}', f'{p}/{new_im_name}')


def im_id_format(path):
    """
    Reformat all images in the given directory so that the id of the person is different according to the different
    days.
    """
    for im in os.listdir(os.path.join(path)):
        day_num = im.split('_')[-1][1:3]
        new_id_name = day_num + im[2:]

        os.rename(f'{path}/{im}', f'{path}/{new_id_name}')


def remove_images_from_dataset(path, pattern):
    """
    Remove images that are matching the given pattern from a dataset folder.
    """
    for im in os.listdir(path):
        if pattern in im:
            os.remove(os.path.join(path, im))


def read_labeled_croped_images(file_dir, file_type='jpg') -> dict:
    """
    used to load images from a folder, recursively and returns dict with mapping between
    each image crop and it's id
    file-dir: dir to load images from
    return an ID to images dict.
    """

    # '/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned' cur path for labled data

    assert os.path.isdir(file_dir), 'Read labeled data must get a valid dir path'
    imgs = defaultdict(list)
    print('reading imgs from dir...')
    for img in tqdm(Path(file_dir).rglob(f'*.{file_type}')):
        img = str(img)
        if not os.path.isfile(img):
            continue
        im_path = os.path.split(img)[1]
        img_id = im_path[:4]  # xxxx id format
        imgs[img_id].append(imread(img))
    return imgs


def trim_videos_from_dir(dir, output_path, limit, create_every=45000):
    for i, vid in enumerate(os.listdir(dir)):
        print("Creating trimmed vid ", i)
        cur_out = os.path.join(output_path, vid[:-4])
        os.makedirs(cur_out, exist_ok=True)
        trim_video(os.path.join(dir, vid), cur_out, limit, create_every)


def trim_video(input_path, output_path, limit, create_every=40000):
    """
    Given a path to a video and the number of frames that should be taken from it,
    trim the video to the first `limit` frames. Saves the output to the `output_path` location.
    by default creates a video every 45K frames, e.g. ~Half an hour
    """
    print(" Loading video to imgs...")
    imgs = mmcv.VideoReader(input_path)
    print(f' Loaded {len(imgs)}')
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    fps = int(imgs.fps)
    print(" Starting trim iter")
    os.makedirs(output_path, exist_ok=True)
    for batch in range(0, len(imgs), create_every):
        cur_imgs = imgs[batch:min(batch + limit + 1, len(imgs))]
        for i, img in enumerate(cur_imgs):
            mmcv.imwrite(img, f'{temp_path}/{i:03d}.png')
            if not i % 100:
                print(f'    {i} frames done.')
            if i == limit:
                print(f'    Captured {limit} frames. Creating vid...')
                vid_name = os.path.split(output_path)[-1]
                mmcv.frames2video(temp_path,
                                  os.path.join(output_path, f'{vid_name}_s{batch}_e{batch + len(cur_imgs)}.mp4'),
                                  fps=fps, fourcc='mp4v', filename_tmpl='{:03d}.png')
                temp_dir.cleanup()
                print('     done.')
                break
        print(' Cleaning up residue..')
        temp_dir.cleanup()
    print(" Done trimming vid.")


def plot_tracks(img, bboxes, labels, ids, masks=None, classes=None, score_thr=0.0, thickness=2, font_scale=0.4,
                show=False, wait_time=0, out_file=None, bbox_colors=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[inds]
    labels = labels[inds]
    ids = ids[inds]
    if masks is not None:
        assert masks.ndim == 3
        masks = masks[inds]
        assert masks.shape[0] == bboxes.shape[0]

    text_width, text_height = 9, 13
    for i, (bbox, label, id, bbox_color) in enumerate(zip(bboxes, labels, ids, bbox_colors)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        if not bbox_color:
            bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        if classes is not None:
            text += f'|{classes[label]}'
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + 2 * text_height,
        x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + 2 * text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # mask
        if masks is not None:
            mask = masks[i].astype(bool)
            mask_color = np.array(bbox_color, dtype=np.uint8).reshape(1, -1)
            img[mask] = img[mask] * 0.5 + mask_color * 0.5

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def create_bbox_color(crop_props: list) -> list:
    """
    Given a list of crops properties (crop.invalid, crop.vague, crop.reviewed_one) for each crop, return a color for
    a bbox according to the following convention:
        - RED: crop was marked as discarded
        - BLUE: crop was marked as vague
        - YELLOW: crop was reviewed yet
        - GREEN: crop was reviewed and wasn't marked as invalid or vague.
    @param crop_props: a list of dictionaries for each crop. Each dictionary is of the form: {'invalid':
    crop.invalid, 'vague': crop.vague, 'reviewed_1': crop.reviewed_one}

    @return a list with the bbox_color string for each crop
    """
    bbox_colors = []
    for crop in crop_props:
        bbox_color = 'yellow'  # yellow, if crop wasn't review yet
        if crop.get('reviewed_1'):  # green, if crop was reviewed and not marked as invalid / vague
            bbox_color = 'green'
        if crop.get('vague'):  # blue, if crop was marked as vague
            bbox_color = 'blue'
        if crop.get('invalid'):  # red, if crop was marked as invalid
            bbox_color = 'red'
        bbox_colors.append(bbox_color)
    return bbox_colors

def create_bbox_color_for_eval(crops):
    bbox_colors = []
    for inference_crop in crops:
        try:
            db_crop = get_entries(filters={Crop.im_name == inference_crop.im_name}).all()[0] # using the tagged DB!
            if db_crop.invalid:
                bbox_color = 'blue'
            elif db_crop.label == inference_crop.label: # compare between db crop and inference crop
                bbox_color = 'green'
            else:
                bbox_color = 'red'
            bbox_colors.append(bbox_color)
        except:
            bbox_colors.append('blue')
    return bbox_colors




def viz_data_on_video_using_pickle(input_vid, output_path, pre_labeled_pkl_path=None, path_to_crops=None):
    """
    This func assumes that the input video has been run by the track and reid model data creator to
    create a pre-annoted set.
    Args:
        input_vid:
        pre_labeled_pkl_path:

    Returns:

    """
    assert pre_labeled_pkl_path or path_to_crops, "You must enter either a pkl to cropsDB or the crops folder"
    crops = None
    if path_to_crops:
        assert os.path.isdir(path_to_crops), "Path must be a CropDB folder"
        crops = []
        for file in os.listdir(path_to_crops):
            crop_path = os.path.join(path_to_crops, file)
            _, extention = os.path.splitext(crop_path)
            if os.path.split(crop_path)[-1][0] != 'Face' and extention in ['.jpg', '.png']:  # skip face crops for viz
                crops.append(create_Crop_from_str(crop_path))
    elif pre_labeled_pkl_path:
        assert os.path.isfile(pre_labeled_pkl_path), "Path must be a CropDB file"
        crops = pickle.load(open(pre_labeled_pkl_path, 'rb'))

    # create frame_to_crops dict
    crop_dict_by_frame = defaultdict(list)
    for crop in crops:
        crop_dict_by_frame[crop.frame_id].append(crop)

    imgs = mmcv.VideoReader(input_vid)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    fps = int(imgs.fps)

    for i, frame in enumerate(imgs):
        cur_crops = crop_dict_by_frame.get(i)
        if cur_crops:
            # at least single crop was found in frame
            crops_bboxes = [np.append(crop.bbox, [1]) for crop in
                            cur_crops]  ## adding 1 for keeping up with plot requirements
            crops_labels = [crop.label for crop in cur_crops]
            cur_img = plot_tracks(img=frame, bboxes=np.array(crops_bboxes), ids=np.array(crops_labels),
                                  labels=np.array(crops_labels))
            mmcv.imwrite(cur_img, f'{temp_path}/{i:03d}.png')
        else:
            # no crops detected, write the original frame
            mmcv.imwrite(frame, f'{temp_path}/{i:03d}.png')

    mmcv.frames2video(temp_path, output_path, fps=fps, fourcc='mp4v', filename_tmpl='{:03d}.png')
    temp_dir.cleanup()


# def create_tracklet_hist(pre_labeled_pkl_path):
#     import seaborn as sns
#     from matplotlib import pyplot as plt
#     import pandas as pd
#     crops = pickle.load(open(pre_labeled_pkl_path, 'rb'))
#     crop_dict_by_frame = defaultdict(int)
#     for crop in crops:
#         crop_dict_by_frame[crop.track_id] += 1
#
#     df = pd.DataFrame({'Track Length': crop_dict_by_frame.values()})
#     sns.histplot(data=df, bins=50)
#     # plt.xticks(list(range(0,501,10)))
#     plt.xlabel("Number of crops in track")
#     plt.ylabel("Count of tracks")
#     plt.title('Track Count of 500 frame video')
#     plt.show()


def viz_DB_data_on_video(input_vid, output_path, db_path=DB_LOCATION_TEST, eval=False):
    """
    Use the labeled data from the DB to visualize the labels on a given video.
    Args:
        - input_vid: the video that should be visualized. NOTE: the DB will be queried according to this video name!
        - output_path: the path in which the labeled output video should be created.
        - DB_path: the path to the DB that holds the labeled crops of the video.
        - eval: use this for inference only. if data is tagged by DB bboxes color will be adapted
    """
    vid_name = input_vid.split('/')[-1][:-4]

    imgs = mmcv.VideoReader(input_vid)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    fps = int(imgs.fps)

    for i, frame in tqdm(enumerate(imgs), total=len(imgs)):
        # retrieve all crops of the current frame from the DB:
        session = create_session(db_path)
        frame_crops = get_entries(session=session, filters=(Crop.vid_name == vid_name, Crop.frame_num == i)).all()
        if frame_crops:
            # at least single crop was found in frame
            crops_bboxes = [np.array([crop.x1, crop.y1, crop.x2, crop.y2, crop.conf]) for crop in frame_crops]
            crops_labels = [crop.label for crop in frame_crops]
            if eval:
                bbox_colors = create_bbox_color_for_eval([crop for crop in frame_crops])
            else:
                bbox_colors = create_bbox_color([{'invalid': crop.invalid, 'vague': crop.is_vague, 'reviewed_1': crop.reviewed_one} for crop in frame_crops])
            bbox_colors_RGB = [COLOR_TO_RGB[bbox] for bbox in bbox_colors]
            cur_img = plot_tracks(img=frame, bboxes=np.array(crops_bboxes), ids=np.array(crops_labels),
                                  labels=np.array(crops_labels), bbox_colors=bbox_colors_RGB)
            mmcv.imwrite(cur_img, f'{temp_path}/{i:03d}.png')
        else:
            # no crops detected, write the original frame
            # print('O NOO')
            mmcv.imwrite(frame, f'{temp_path}/{i:03d}.png')

    print(f'Saving video into: {output_path}')
    mmcv.frames2video(temp_path, output_path, fps=fps, fourcc='mp4v', filename_tmpl='{:03d}.png')
    temp_dir.cleanup()


def build_samples_hist(title:str=None):
    crops = get_entries(filters=(Crop.invalid == False, Crop.reviewed_one == True))
    track_counter = defaultdict(set)
    for crop in crops:
        # if '20210804' not in crop.vid_name:
        #     continue
        # if '20210804' in crop.vid_name :
        track_counter[crop.label].add(str(crop.vid_name) +'_' + str(crop.track_id))

    ret = {}
    for k,v in track_counter.items():
        ret[k] = len(v)

    """Create a Bar plot that represented the number of face samples from every id"""
    # cnt = {ID_TO_NAME[int(k)] : [len(samples_dict[k])] for k in samples_dict.keys()}
    df = pd.DataFrame(ret, index=ret.keys())
    ax= sns.barplot(data=df)
    ax.set_xticklabels(df.columns, rotation=45)
    plt.title(title)
    plt.show()

# def main():
#     """Simple test of FaceDetector"""
#     print('hoi')
#     crops = get_entries(filters=({Crop.invalid == False, Crop.reviewed_one == True}))
#     track_counter = defaultdict(set)
#     for crop in crops:
#         track_counter[crop.label].add(str(crop.vid_name) +'_' + str(crop.track_id))
#
#     ret = {}
#     for k,v in track_counter.items():
#         ret[k] = len(v)


if __name__ == '__main__':
    # build_samples_hist()
    trim_video("/mnt/raid1/home/bar_cohen/42street/output002.mp4", "/mnt/raid1/home/bar_cohen/42street/train_videos_2/",limit=500,create_every=500)
    # trim_videos_from_dir(dir="/mnt/raid1/home/bar_cohen/42street/val_videos_3/", output_path='/mnt/raid1/home/bar_cohen/42street/val_videos_3/',
    #                      limit=500, create_every=1000)
    # viz_data_on_video_using_pickle(input_vid='/home/bar_cohen/KinderGuardian/Videos/trimmed_1.8.21-095724.mp4',
    #                   output_path="/home/bar_cohen/KinderGuardian/Results/trimmed_1.8.21-095724_labled1.mp4",
    #                   pre_labeled_pkl_path='/mnt/raid1/home/bar_cohen/DB_Crops/_crop_db.pkl')
    # path_to_crops="/mnt/raid1/home/bar_cohen/DB_Crops/")

    # im_name_format('/home/bar_cohen/D-KinderGuardian/fast-reid/datasets/2.8.21-dataset/query')

    # Data Visualization:
    # vid_name = '20210804151703_s45000_e45501'
    # vid_date = vid_name.split('_')[0]
    # kinder_guardian_path = '/home/bar_cohen/KinderGuardian'
    # os.makedirs(f'{kinder_guardian_path}/DataProcessing/Data/_{vid_name}/', exist_ok=True)
    # viz_DB_data_on_video(
    #     input_vid=f'/home/bar_cohen/raid/trimmed_videos/IPCamera_{vid_date}/IPCamera_{vid_name}.mp4',
    #     output_path=f'{kinder_guardian_path}/DataProcessing/Data/_{vid_name}/{vid_name}_reviewed1.mp4')
    #     # output_path=f'/home/bar_cohen/D-KinderGuardian/DataProcessing/Data/_{vid_name}/{vid_name}.mp4')

    # Duke Video Dataset Rename:
    # im_name_format('/mnt/raid1/home/bar_cohen/OUR_DATASETS/DukeMTMC-VideoReID/query')

