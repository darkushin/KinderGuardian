import logging

import mmcv
import numpy as np
import torch
from mmtrack.apis import inference_mot, init_model
import os, glob, shutil
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2

class Crop:
    def __init__(self, frame_id:int,
                 bbox:np.array,
                 crop_img:torch.tensor,
                 face_img:torch.tensor,
                 track_id:int,
                 cam_id:int,
                 crop_id:int,
                 video_name:str):
        self.frame_id = int(frame_id)
        self.bbox = bbox
        self.crop_img = crop_img
        self.face_img = face_img
        self.track_id = int(track_id)
        self.cam_id = int(cam_id)
        self.crop_id = int(crop_id)
        self.video_name = video_name
        self.label = None
        self.unique_crop_name = None
        self.update_hash()

    def set_label(self, label):
        self.label = label
        self.update_hash() # update unique_hash

    def update_hash(self):
         # save format - f'video_name:{self.video_name[9:]}_track_id:{self.track_id}_cam_id:{self.cam_id}_frame_id:{self.frame_id}_crop_id:{self.crop_id}_label:{self.label}'
        self.unique_crop_name = f'{self.label}_v{self.video_name[9:]}_f{self.frame_id}_b{str(self.bbox)}_t{self.track_id}_c{self.cam_id}_cid{self.crop_id}'

    def save_crop(self, datapath):
        mmcv.imwrite(self.crop_img, os.path.join(datapath, f'{self.unique_crop_name}.png'))
        if self.check_if_face_img():
            face_to_write = self.face_img.permute(1, 2, 0).int().numpy()
            mmcv.imwrite(face_to_write, os.path.join(datapath, f'Face_{self.unique_crop_name}.png'))

    def check_if_face_img(self):
        return self.face_img is not None and self.face_img is not self.face_img.numel()

def create_Crop_from_str(img_path):
    # self.unique_crop_name = f'{self.label}_v{self.video_name[9:]}_f{self.frame_id}_b{str(self.bbox)}_t{self.track_id}_c{self.cam_id}_cid{self.crop_id}'
    splitted = img_path.split('_')
    label = splitted[0]
    video_name = splitted[1][1:]
    frame_id = splitted[2][1:]
    bbox = list(splitted[3][1:])
    track_id = splitted[4][1:]
    cam = splitted[5][1:]
    cid = splitted[6][1:]
    crop_obj = Crop(frame_id=frame_id,bbox=bbox,crop_img=None,face_img=None,
                track_id=track_id,cam_id=cam,crop_id=cid, video_name=video_name)
    crop_obj.set_label(label)
    return crop_obj

class DataHandler:

    def __init__(self,
                 input_folder_path: str,
                 output_video_path: str,
                 config,
                 checkpoint,
                 device,
                 ):
        """
        Creates the DataHandler Obj, in charge of holding the Data Processing module capabilities.
        Assumes input was validated by DataFactory/ScriptRunner.
        @param input_folder_path: Input folder for videos to parse. must be non-empty with avi format videos
        @param output_video_path: Output Folder for crops and clusters
        @param config: the config file path for the mm-track module
        @param checkpoint: the checkpoint file path for the mm-track module, if non entered downloads checkpoint during runtime.
        @param device: the device to run script from. Defaults to GPU if one exists.
        """

        self.input_path = input_folder_path
        self.output_path = output_video_path
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.crop_folder_path = os.path.join(self.output_path, 'crops')
        self.cluster_folder_path = os.path.join(self.output_path + 'clusters')
        self.cam_id = 2  # this is a remaining issue , we need to parse video input and extract cam id

    def track_persons_and_crop_from_dir(self, capture_index: int = 500, acc_threshold: int = 0.999) -> None:
        """
        For each video in input directory track persons and create a cropped image based on every 'capture_index'
        frame, if the confidence of a person detected in a crop exceeds acc_threshold param.
        @param capture_index: a frame will be captured and analyzed was capture_index frames have passed
        @param acc_threshold: a crop will be saved only if the model's confidence is above acc_treshold
        @return: None.
        """
        """
        capture_index = frame will be captured every capture_index value. input 1 for all frames
        """
        logging.info("Load mm-track model")
        model = init_model(self.config, self.checkpoint, device=self.device)
        os.makedirs(self.crop_folder_path, exist_ok=True)
        if os.path.isdir(self.input_path):
            vids = os.listdir(self.input_path)
            assert len(vids) > 0, "Input folder must be non-empty"
            print(f'Found the following videos to process: {vids}')
            for vid in vids:
                cur_path = os.path.join(self.input_path, vid)
                if not os.path.isdir(cur_path) and cur_path.endswith(('.mp4', '.avi')):
                    print(f'Processing video: {cur_path}')
                    imgs = mmcv.VideoReader(cur_path)
                    print('begin extraction of video {}'.format(vid))
                    prog_bar = mmcv.ProgressBar(len(imgs))
                    # test and show/save the images
                    for i, img in enumerate(imgs):
                        if i % capture_index != 0:
                            # we only capture every capture_index frames
                            prog_bar.update()
                            continue

                        if isinstance(img, str):
                            img = os.path.join(self.input_path, img)
                        prog_bar.update()
                        result = inference_mot(model, img, frame_id=i)  # using mmtrack inference
                        acc = result['track_results'][0][:, -1]
                        ids = result['track_results'][0][:, 0]
                        mask = np.where(acc > acc_threshold)  # accepting accuracy above threshold
                        croped_boxes = result['bbox_results'][0][:, :-1]
                        croped_boxes = croped_boxes[mask]
                        croped_im = mmcv.image.imcrop(img, croped_boxes, scale=1.0, pad_fill=None)
                        ids = ids[mask]

                        for j in range(len(croped_im)):
                            per_crop = np.array(croped_im[j])
                            # data set example "0005_c2_f0046985.jpg"
                            # add the name of the video to avoid overwriting of crops with the same name from different videos:
                            vid_suffix = vid.split('.avi')[0]
                            cropped_out = f'{self.crop_folder_path}/{int(ids[j]):04d}_c{self.cam_id}_f{i:07d}_vid-{vid_suffix}.jpg'
                            cv2.imwrite(cropped_out, per_crop)
                        prog_bar.update()
                else:
                    logging.warning(f"Only avi or mp4 files are supported ; skipping file {cur_path}")
        else:
            raise Exception(f'Unsupported Input Folder! Input path must be a non-empty folder with videos')

    def create_clusters(self, k=10):
        """
        Based on crops saved on the output path folder, run clustering to unsupervised-ly label the Ids
        @param k: K clusters to create
        @return: None
        """
        # we can extend this later to receive input from args
        # if generating the crops in the same run, the input for the clustering should be the output arg

        os.makedirs(self.cluster_folder_path, exist_ok=True)
        joined_path_to_files = os.path.join(self.crop_folder_path,'*.*')
        images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(joined_path_to_files)]
        paths = [file for file in glob.glob(joined_path_to_files)]
        assert images and paths , "crops folder must be non-empty"
        images = np.array(np.float32(images).reshape(len(images), -1) / 255)
        model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        predictions = model.predict(images.reshape(-1, 224, 224, 3))
        pred_images = predictions.reshape(images.shape[0], -1)
        kmodel = KMeans(n_clusters=k, random_state=728)
        kmodel.fit(pred_images)
        kpredictions = kmodel.predict(pred_images)
        for i in range(k):
            os.makedirs(f'{self.cluster_folder_path}/cluster{str(i)}', exist_ok=True)
        for i in range(len(paths)):
            shutil.copy2(paths[i], f'{self.cluster_folder_path}/cluster{str(kpredictions[i])}')