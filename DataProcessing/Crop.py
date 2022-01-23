import os
import torch
import numpy as np


class Crop:
    def __init__(self, frame_id: int,
                 bbox: np.array,
                 crop_img: torch.tensor,
                 face_img: torch.tensor,
                 track_id: int,
                 cam_id: int,
                 crop_id: int,
                 video_name: str,
                 is_face : bool
                 ):
        self.frame_id = int(frame_id)
        self.bbox = bbox
        self.crop_img = crop_img
        self.face_img = face_img
        self.track_id = int(track_id)
        self.cam_id = int(cam_id)
        self.crop_id = int(crop_id)
        self.video_name = video_name # note the name must be already 'clean' prior to crop Init
        self.label = None
        self.unique_crop_name = None
        self.update_hash()
        self.is_face = is_face
        self.is_vague = False

    def set_label(self, label):
        self.label = label

    def update_hash(self):
        self.unique_crop_name = f'v{self.video_name}_f{self.frame_id}_b{str(list(self.bbox))}'

    def save_crop(self, datapath):
        mmcv.imwrite(self.crop_img, os.path.join(datapath, f'{self.unique_crop_name}.png'))
        # if self.check_if_face_img():
        #     face_to_write = self.face_img.permute(1, 2, 0).int().numpy()
        #     mmcv.imwrite(face_to_write, os.path.join(datapath, f'Face_{self.unique_crop_name}.png'))

    def check_if_face_img(self):
        self.is_face = self.face_img is not None and self.face_img is not self.face_img.numel()
        return self.is_face

    def manually_set_is_face(self, is_face):  # used for deserializing Crop obj only
        self.is_face = is_face
