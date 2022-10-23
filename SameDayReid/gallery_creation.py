import pickle
import shutil
import sys
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import mmcv
import os
from DataProcessing.DB.dal import add_entries, SAME_DAY_DB_LOCATION, get_entries, create_session, \
    SameDayCropV2
from DataProcessing.dataProcessingConstants import ID_TO_NAME

DB_GALLERY = "/mnt/raid1/home/bar_cohen/42street/db_gallery/"
LABELED_TRACK_GALLERY = "/mnt/raid1/home/bar_cohen/42street/labeled_track_gallery_t/"


sys.path.append('FaceDetection')

from FaceDetection.arcface import ArcFace, GALLERY_PKL_PATH, GPIDS_PKL_PATH, is_img, GPATH

sys.path.append('mmpose')
import tqdm
import numpy as np
from FaceDetection.pose_estimator import PoseEstimator
from mmtracking.mmtrack.apis import init_model, inference_mot
from mmtrack.apis import inference_mot, init_model

TRACKING_CONFIG_PATH = "/home/bar_cohen/KinderGuardian/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
TRACKING_CHECKPOINT = "/home/bar_cohen/KinderGuardian/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
FACE_CHECKPOINT = "/mnt/raid1/home/bar_cohen/FaceData/checkpoints/4.8 Val, 1.pth"
POSE_CONFIG = "/home/bar_cohen/D-KinderGuardian/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
POSE_CHECKPOINT =  "/home/bar_cohen/D-KinderGuardian/checkpoints/mmpose-hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
FACE_NET = 'FaceNet'
ARC_FACE = 'ArcFace'
FOLDER_HIERARCHY = 'folder_hierarchy'
ENRICH_ENRICHED = 'enrich_enriched'
MIN_IMG_SIZE_FOR_DOWNSAMPLE = 300
DOWN_SAMPLE_RATIO = 0.2

class GalleryCreator:
    def __init__(self, gallery_path,
                 tracker_conf_threshold = 0.99,
                 similarty_threshold = 0.5,
                 device= 'cuda:0',
                 cam_id = '1',
                 track_config=TRACKING_CONFIG_PATH,
                 track_checkpoint=TRACKING_CHECKPOINT,
                 create_in_fastreid_format=False,
                 init_models = True
                 ):
        self.similarty_threshold = similarty_threshold
        self.tracking_model = init_model(track_config, track_checkpoint, device=device)
        self.tracker_conf_threshold = tracker_conf_threshold
        self.cam_id = cam_id
        if create_in_fastreid_format:
            os.makedirs(gallery_path)
            os.makedirs(os.path.join(gallery_path,'bounding_box_test'))
            os.makedirs(os.path.join(gallery_path,'bounding_box_train'))
            os.makedirs(os.path.join(gallery_path,'query'))
            self.gallery_path = os.path.join(gallery_path,'bounding_box_test')
        else:
            os.makedirs(gallery_path, exist_ok=True)
            self.gallery_path = gallery_path
        self.global_i = 0

        if init_models:
            self.arc = ArcFace(gallery_path=GPATH)
            # self.arc.read_gallery_from_scratch()
            # self.arc.save_gallery_to_pkl(GALLERY_PKL_PATH, GPIDS_PKL_PATH)
            self.arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH, gpid_path=GPIDS_PKL_PATH)
            self.pose_estimator = PoseEstimator(pose_config=POSE_CONFIG, pose_checkpoint=POSE_CHECKPOINT, device=device) # TODO hard code configs here

    def get_vid_name_from_path(self, video_path):
        return ''.join(video_path.split(os.sep)[-2:])

    def get_42street_part(self, video_path):
        return int(video_path.split(os.sep)[-2][-1])

    def add_video_to_db(self, video_path:str, skip_every=1, db_location=SAME_DAY_DB_LOCATION):
        imgs = mmcv.VideoReader(video_path)
        vid_name = self.get_vid_name_from_path(video_path=video_path)
        part = self.get_42street_part(video_path=video_path) # 42street specific
        video_crops = []
        print('Detecting faces and creating DB type Crop objects...')
        for image_index, img in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
            if image_index % skip_every != 0:
                continue
            result = tracking_inference(self.tracking_model, img, image_index,
                                        acc_threshold=float(self.tracker_conf_threshold))
            ids = list(map(int, result['track_results'][0][:, 0]))
            confs = result['track_results'][0][:, -1]
            crops_bboxes = result['track_results'][0][:, 1:-1]
            crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
            video_crops.extend(self.detect_faces_and_create_db_crops(ids=ids,
                                                        confs=confs,
                                                        crops_imgs=crops_imgs,
                                                        crops_bboxes=crops_bboxes,
                                                        vid_name=vid_name,
                                                        part=part,
                                                        frame_num=image_index
                                                        ))

        add_entries(crops=video_crops, db_location=db_location) # Note this points to the SAME_DB_DB_LOCATION !
        
    def detect_faces_and_create_db_crops(self, ids, confs, crops_imgs, crops_bboxes, vid_name, part, frame_num):
        crops = []
        pose_discard_counter = 0
        errs_counter = 0
        for i, (id, conf, crop_im, crop_bbox) in enumerate(zip(ids, confs, crops_imgs, crops_bboxes)):
            # try:
            ## first detection run --- detect all faces in image
            face_imgs, face_bboxes, face_probs = self.arc.detect_face_from_img(crop_img=crop_im)
            # if only a single face exists, apply a high threshold to get a high resolution face image and body
            if face_imgs is not None and len(face_imgs) == 1:
                face_img, face_prob = self.pose_estimator.find_matching_face(crop_im, face_bboxes, face_probs,
                                                                             face_imgs)
                if is_img(face_img) and face_prob >= 0.5:  # hard coded min threshold for face rec, as the default
                    x1_crop, y1_crop, x2_crop, y2_crop = list(map(int, crops_bboxes[i]))  # convert the bbox floats to ints
                    x_face, y_face, w_face, h_face = list(map(int,face_bboxes[0])) # convert the bbox floats to ints
                    crop = SameDayCropV2(
                                vid_name=vid_name,
                                part=part,
                                gt_label='',
                                label='',
                                im_name='',
                                face_im_name='',
                                frame_num=frame_num,
                                x1_crop=x1_crop,
                                y1_crop=y1_crop,
                                x2_crop=x2_crop,
                                y2_crop=y2_crop,
                                x_face=x_face,
                                y_face=y_face,
                                w_face=w_face,
                                h_face=h_face,
                                face_conf=face_prob,
                                face_cos_sim=-1,
                                face_ranks_diff=-1,
                                track_id=id,
                                cam_id=5, # a constant 5 for now
                                crop_id=i
                    )
                    crop.set_im_name()
                    crops.append(crop)
                    #Note - writing crop to disk prior to writing in DB
                    mmcv.imwrite(np.array(crop_im), os.path.join(DB_GALLERY, crop.im_name))
                else:
                    pose_discard_counter += 1
        return crops

    def label_video(self, vid_name:str):
        def score_boundary(min_score_threshold:float):
            cur_diff = ranks[0] - ranks[1]
            assert cur_diff > 0
            return cur_diff

        session = create_session(db_location=SAME_DAY_DB_LOCATION)
        crops = get_entries(session=session,
                            filters=({SameDayCropV2.vid_name == self.get_vid_name_from_path(video_path=vid_name)}),
                            db_path=SAME_DAY_DB_LOCATION,
                            crop_type=SameDayCropV2).all()
        crop_faces = [mmcv.imread(os.path.join(DB_GALLERY, crop.im_name))[crop.y_face:crop.h_face, crop.x_face:crop.w_face] for crop in crops]
        if len(crop_faces) > 0:  # some faces where detected
            for crop_obj , face in tqdm.tqdm(zip(crops, crop_faces), total=len(crops)):
                face = face[:, :, ::-1] # switch color channels
                if face.shape[0] > 0 and face.shape[1] > 1 and crop_obj.face_conf >= 0.8:
                    cur_score = self.arc.predict_img(face)
                    ranks = sorted(cur_score.values(), reverse=True)
                    label = max(cur_score, key=cur_score.get)
                    diff = score_boundary(min_score_threshold=0.5)
                    print(f"diff is {(ranks[0] - ranks[1])}, max score is {max(cur_score.values())}, "
                          f"face conf is {crop_obj.face_conf}, the given label is {label}")
                    crop_obj.label = label
                    crop_obj.face_cos_sim = float(max(cur_score.values()))
                    crop_obj.face_ranks_diff = diff
        session.commit()

    def add_video_to_gallery_from_same_day_DB(self, vid_name:str, face_conf_threshold:float, face_sim_threshold:float,
                                              min_ranks_diff_threshold:float, create_labeled_training=False,
                                              augment=False):

        session = create_session(db_location=SAME_DAY_DB_LOCATION)
        good_crops = get_entries(session=session,
                            filters=({SameDayCropV2.vid_name == self.get_vid_name_from_path(video_path=vid_name),
                                      SameDayCropV2.face_conf >= face_conf_threshold,
                                      SameDayCropV2.face_cos_sim >= face_sim_threshold,
                                      SameDayCropV2.face_ranks_diff >= min_ranks_diff_threshold,
                                      }
                                      ),
                            db_path=SAME_DAY_DB_LOCATION,
                            crop_type=SameDayCropV2).all()

        if not create_labeled_training:
            unknown_crops = get_entries(session=session,
                                filters=({SameDayCropV2.vid_name == self.get_vid_name_from_path(video_path=vid_name),
                                          SameDayCropV2.face_conf > 0.87,
                                          SameDayCropV2.face_cos_sim < 0.30,
                                          SameDayCropV2.face_ranks_diff < 0.01,
                                          }
                                          ),
                                db_path=SAME_DAY_DB_LOCATION,
                                crop_type=SameDayCropV2).all()

            for crop in unknown_crops:
                crop.label = 12

            good_crops.extend(unknown_crops)

        for crop in tqdm.tqdm(good_crops):
            if crop.label:
                crop_name = f'{int(crop.label):04d}_c{self.cam_id}_f{self.global_i:07d}.jpg'
                shutil.copy(os.path.join(DB_GALLERY, crop.im_name), os.path.join(self.gallery_path, crop_name))
                if create_labeled_training:
                    # save the face crop as well as the full body crop
                    crop_face = mmcv.imread(os.path.join(DB_GALLERY, crop.im_name))[crop.y_face:crop.h_face,
                                 crop.x_face:crop.w_face]
                    crop_face_name = f'{int(crop.label):04d}_c{self.cam_id}_f{self.global_i:07d}_face.jpg'
                    mmcv.imwrite(np.array(crop_face), os.path.join(self.gallery_path, crop_face_name))
                self.global_i += 1
                if augment:
                    img = mmcv.imread(os.path.join(DB_GALLERY, crop.im_name))
                    crop_name = f'{int(crop.label):04d}_c{self.cam_id}_f{self.global_i:07d}.jpg' # updated global_i
                    downsampled = augment_by_downsampling(img)
                    if downsampled:
                        mmcv.imwrite(np.array(downsampled), os.path.join(self.gallery_path, crop_name))
                    self.global_i += 1

    def create_labeled_tracks_using_DB(self, video_path,save_type,augment=False):
        assert save_type in [FOLDER_HIERARCHY, ENRICH_ENRICHED] , "Save Type param not supported"
        session = create_session(db_location=SAME_DAY_DB_LOCATION)
        imgs = mmcv.VideoReader(video_path)
        vid_name = self.get_vid_name_from_path(video_path=video_path)
        part = self.get_42street_part(video_path=video_path) # 42street specific
        track_label_dict = dict()
        track_imgs = defaultdict(list)
        for image_index, img in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
            result = tracking_inference(self.tracking_model, img, image_index,
                                        acc_threshold=float(self.tracker_conf_threshold))
            ids = list(map(int, result['track_results'][0][:, 0]))
            crops_bboxes = result['track_results'][0][:, 1:-1]
            crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
            for crop_img,track_id in zip(crops_imgs, ids):
                    if track_id not in track_label_dict.keys():
                        labeled_track_crops = get_entries(session=session, filters={SameDayCropV2.vid_name == vid_name,
                                                                                    SameDayCropV2.track_id == track_id,
                                                                                    SameDayCropV2.part == part,
                                                                                    SameDayCropV2.face_conf >= 0.83,
                                                                                    SameDayCropV2.face_cos_sim >= 0.6,
                                                                                    }, crop_type=SameDayCropV2, db_path=SAME_DAY_DB_LOCATION).all()

                        if len(labeled_track_crops) > 10:
                            track_labels = [crop.label for crop in labeled_track_crops]
                            bincount = np.bincount(track_labels)
                            track_maj_vote = np.argmax(bincount)
                            track_maj_conf = bincount[track_maj_vote] / len(labeled_track_crops)
                            if track_maj_conf > 0.95:
                                track_label_dict[track_id] = track_maj_vote

                        else:
                            track_label_dict[track_id] = -1 # this track does not have what it takes, skip it on the next img
                    if track_id in track_label_dict.keys() and track_label_dict[track_id] != -1:
                        track_imgs[track_id].append(crop_img)

        for track_id in track_imgs.keys():
            if track_label_dict[track_id] != -1: # this track has high quality faces labeled
                for i,crop_im in enumerate(track_imgs[track_id]):
                    cur_path, crop_name = None, None
                    cur_label = int(track_label_dict[track_id])
                    if save_type == FOLDER_HIERARCHY:
                        cur_path = os.path.join(LABELED_TRACK_GALLERY,
                                                f"{cur_label:04d}",
                                                f"v{self.global_i:03d}_t{track_id:07d}")
                        os.makedirs(cur_path, exist_ok=True)
                        crop_name = f'{i:07d}.jpg'
                        mmcv.imwrite(np.array(crop_im), os.path.join(cur_path, crop_name))

                    elif save_type == ENRICH_ENRICHED:
                        cur_path = self.gallery_path
                        crop_name = f'{cur_label:04d}_c{self.cam_id}_f{self.global_i:07d}.jpg'
                        mmcv.imwrite(np.array(crop_im), os.path.join(cur_path, crop_name))
                        self.global_i += 1 # here global i is per crop
                        if augment:
                            downsampled = augment_by_downsampling(crop_im)
                            if downsampled:
                                crop_name = f'{cur_label:04d}_c{self.cam_id}_f{self.global_i:07d}.jpg' #global_i updated
                                mmcv.imwrite(np.array(downsampled), os.path.join(cur_path, crop_name))
                                self.global_i += 1  # here global i is per crop

        if FOLDER_HIERARCHY: # we only distinguish between video
            self.global_i += 1 # this will be used for diff videos

def augment_by_downsampling(img):
    resize_img = None
    img = Image.fromarray(img)
    if min(img.size) >= MIN_IMG_SIZE_FOR_DOWNSAMPLE:
        resize_img = img.resize((int(img.size[0]*DOWN_SAMPLE_RATIO), int(img.size[1]*DOWN_SAMPLE_RATIO)))
    return resize_img

def is_high_quality_unknown(crop:SameDayCropV2):
    # the difference between rank1 and rank2 is low, the max score received for known ids is low but the face
    # itself is of high quality --- possibly an unknown?
    return crop.face_ranks_diff <= 0.01 and crop.face_cos_sim <= 0.30 and crop.face_conf >= 0.87

def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


if __name__ == '__main__':
    print("Thats right yall")
    sample_img = "/mnt/raid1/home/bar_cohen/42street/temp/0003_c5_f0000830.jpg"
    img = mmcv.imread(sample_img)
    #
    gc = GalleryCreator(gallery_path="/mnt/raid1/home/bar_cohen/42street/part2_track_enriched_down_sampled/", cam_id="5",
                        device='cuda:0', create_in_fastreid_format=True, tracker_conf_threshold=0.0, init_models=False)
    print('Done Creating Gallery pkls')
    vid_path = "/mnt/raid1/home/bar_cohen/42street/val_videos_2/"
    vids = [os.path.join(vid_path, vid) for vid in os.listdir(vid_path)]
    for i,vid in enumerate(vids):
        # if i > 1:
        #     break
        print(f'doing vid {i} out of {len(vids)}')
        # gc.add_video_to_db(vid, skip_every=1)
        # if "_s21500_e22001.mp4" not in vid:
        #     continue
        # print(f'Labeling video. {vid}')
        # gc.label_video(vid_name=vid)
        print('Adding labeled images to gallery')
        gc.add_video_to_gallery_from_same_day_DB(vid_name=vid, face_conf_threshold=0.80,
                                                 face_sim_threshold=0.5,
                                                 min_ranks_diff_threshold=0.1,
                                                 create_labeled_training=False, augment=True)
        print('Adding high confidence labeled tracks to gallery')
        gc.create_labeled_tracks_using_DB(vid, save_type=ENRICH_ENRICHED, augment=True)



