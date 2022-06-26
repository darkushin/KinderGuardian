import os

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo
from PIL import Image, ImageDraw
import numpy as np
from FaceDetection.faceDetector import FaceDetector
import torchvision

KEYPOINTS_TO_CHECK = ['nose', 'left_eye', 'right_eye']


class PoseEstimator:
    def __init__(self, pose_config, pose_checkpoint, device):
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.device = device

        # init the model:
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint, device=self.device.lower())
        self.dataset_info = DatasetInfo(self.pose_model.cfg.data['test'].get('dataset_info', None))

    def get_pose(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        np_img = np.array(img)[:, :, ::-1]  # for pose estimation model, image should be a np array and need to swap channels

        # Since we are using a top-down model, we need to provide a bbox of the different persons in the image. At this
        # point, the image we pass to the model is a crop around the person of interest, therefore, the bbox we pass is
        # simply the size of the image. This way the pose model will focus on the main person in the image.
        person_bbox = [{'bbox': [0, 0, np_img.shape[1]-1, np_img.shape[0]-1]}]

        # apply pose model:
        pose_results, _ = inference_top_down_pose_model(self.pose_model, np_img, person_bbox,
                                                        format='xywh', dataset_info=self.dataset_info)

        return pose_results

    def does_face_match_to_pose(self, pose_results, face_bbox):
        does_match = True
        for keypoint in KEYPOINTS_TO_CHECK:
            keypoint_x, keypoint_y, conf = pose_results[0]['keypoints'][self.dataset_info.keypoint_name2id[keypoint]]
            if not all([int(keypoint_x) >= face_bbox[0],
                        int(keypoint_y) >= face_bbox[1],
                        int(keypoint_x) <= face_bbox[2],
                        int(keypoint_y) <= face_bbox[3]]):
                does_match = False
                break
        return does_match

    def find_matching_face(self, img, face_bboxes, face_probs, face_imgs):
        pose_estimation = self.get_pose(img)

        for face_bbox, face_prob, face_img in zip(face_bboxes, face_probs, face_imgs):
            face_bbox = [int(x) for x in face_bbox]
            if self.does_face_match_to_pose(pose_estimation, face_bbox):
                return face_img, face_prob

        # if we reached this section, none of the detected faces in the image matches the pose
        return None, 0

    def visualize_pose(self, img, pose_estimation, face_bboxes=None, output_path=None):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        if face_bboxes is not None:
            for face_bbox in face_bboxes:
                face_bbox = [int(x) for x in face_bbox]
                img1 = ImageDraw.Draw(img)
                img1.rectangle(face_bbox, outline="red")
        np_img = np.array(img)[:, :, ::-1]  # for pose estimation model, image should be a np array and need to swap channels
        vis_img = vis_pose_result(self.pose_model, np_img, pose_estimation, dataset_info=self.dataset_info, out_file=output_path)
        return vis_img


if __name__ == '__main__':
    pose_config = '/home/bar_cohen/D-KinderGuardian/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    pose_checkpoint = '/home/bar_cohen/D-KinderGuardian/checkpoints/mmpose-hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    device = 'cuda:0'
    imgs_path = '/home/bar_cohen/D-KinderGuardian/FaceDetection/crops_with_faces'

    face_detector = FaceDetector(keep_all=True, device=device)
    pose_estimator = PoseEstimator(pose_config, pose_checkpoint, device)

    for img_path in os.listdir(imgs_path):
        img = Image.open(os.path.join(imgs_path, img_path)).convert('RGB')

        face_bboxes, probs = face_detector.facenet_detecor.detect(img=img)
        if face_bboxes is None:
            print(f'No face detected in image {img_path}')
            continue
        pose_estimation = pose_estimator.get_pose(img)

        face_bbox = [int(x) for x in face_bboxes[0]]

        if pose_estimator.does_face_match_to_pose(pose_estimation, face_bbox):
            output_path = os.path.join('/home/bar_cohen/D-KinderGuardian/FaceDetection/results-test/match', img_path)
        else:
            output_path = os.path.join('/home/bar_cohen/D-KinderGuardian/FaceDetection/results-test/no-match', img_path)

        pose_estimator.visualize_pose(img, pose_estimation, face_bbox, output_path)

