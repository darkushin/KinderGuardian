import pickle
import sys

import mmcv

import os

from DataProcessing.DB.dal import SameDayCrop, add_entries, SAME_DAY_DB_LOCATION, get_entries, create_session

sys.path.append('FaceDetection')

from FaceDetection.arcface import ArcFace, GALLERY_PKL_PATH, GPIDS_PKL_PATH, GALLERY_NO_UNKNOWNS, GPIDS_NO_UNKNOWNS, \
    is_img

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
GPATH = "/mnt/raid1/home/bar_cohen/42street/new_face_det_clusters/"


class GalleryCreator:
    def __init__(self, gallery_path,
                 tracker_conf_threshold = 0.99,
                 similarty_threshold = 0.5,
                 device= 'cuda:0',
                 cam_id = '1',
                 track_config=TRACKING_CONFIG_PATH,
                 track_checkpoint=TRACKING_CHECKPOINT,
                 create_in_fastreid_format=False,
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

        self.arc = ArcFace(gallery_path=GPATH)
        # self.arc.read_gallery_from_scratch()
        # self.arc.save_gallery_to_pkl(GALLERY_PKL_PATH, GPIDS_PKL_PATH)
        self.arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH, gpid_path=GPIDS_PKL_PATH)
        self.pose_estimator = PoseEstimator(pose_config=POSE_CONFIG, pose_checkpoint=POSE_CHECKPOINT, device=device) # TODO hard code configs here
        self.global_i = 0

    def get_vid_name_from_path(self, video_path):
        return ''.join(video_path.split('/')[-2:])

    def get_42street_part(self, video_path):
        return int(video_path.split('/')[-2][-1])

    def add_video_to_db(self, video_path:str, skip_every=1):
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

        add_entries(crops=video_crops, db_location=SAME_DAY_DB_LOCATION) # Note this points to the SAME_DB_DB_LOCATION !

            # if label_crops:
            #     self.label_track_according_face_classifier(crops_imgs=crops_imgs, crop_candidates_inds=crop_candidates_inds,
            #                                                crop_candidates_faces=crop_candidates_faces)

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
                    crop = SameDayCrop(
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
                                track_id=id,
                                cam_id=5, # a constant 5 for now
                                crop_id=i
                    )
                    crop.set_im_name()
                    crops.append(crop)
                    #Note - writing crop to disk prior to writing in DB
                    mmcv.imwrite(np.array(crop_im), os.path.join(self.gallery_path, crop.im_name))
                else:
                    pose_discard_counter += 1
            # except Exception as e:
            #     print(e)
            #     errs_counter += 1
            #     continue

        # print(f'Total faces detected in img {len(crops)}/{len(crops_imgs)} crops. \n'
        #       f'A total of {pose_discard_counter} images were discarded due to pose estimation \n'
        #       f'A total of {errs_counter} error were encountered.'
        #       )
        # return crop_candidates_faces, crop_candidates_inds
        return crops

    def label_video(self, vid_name:str, similarity_threshold:float):
        session = create_session(db_location=SAME_DAY_DB_LOCATION)
        crops = get_entries(session=session, filters=({SameDayCrop.vid_name == vid_name}), db_path=SAME_DAY_DB_LOCATION, crop_type=SameDayCrop).all()
        crop_faces = [mmcv.imread(crop.im_name)[crop.y_face:crop.h_face, crop.x_face:crop.w_face] for crop in crops]
        if len(crop_faces) > 0:  # some faces where detected
            for crop_obj , face in zip(crops, crop_faces):
                face = face[:, :, ::-1] # switch color channels
                cur_score = self.arc.predict_img(face)
                label = max(cur_score, key=cur_score.get)
                print(f"the score for the best label match is: {cur_score[label]}")
                if cur_score[label] >= similarity_threshold:  # HIGH THRESHOLD for ReiD
                    crop_obj.label = label
        session.commit()

    def add_video_to_gallery_from_same_day_DB(self, video_name):
        pass
                    # crop_name = f'{label:04d}_c{self.cam_id}_f{self.global_i:07d}.jpg'
                    # self.global_i += 1
                    # mmcv.imwrite(np.array(crop_im), os.path.join(self.gallery_path, crop_name))
#
def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


if __name__ == '__main__':
    print("Thats right yall")
    gc = GalleryCreator(gallery_path="/mnt/raid1/home/bar_cohen/42street/db_gallery/", cam_id="5",
                        device='cuda:1', create_in_fastreid_format=False, tracker_conf_threshold=0.0)
    # print('Done Creating Gallery pkls')
    vid_path = "/mnt/raid1/home/bar_cohen/42street/val_videos_2/"
    vids = [os.path.join(vid_path, vid) for vid in os.listdir(vid_path)]
    for vid in vids:
        gc.add_video_to_db(vid, skip_every=1)

