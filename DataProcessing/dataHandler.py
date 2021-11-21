import os.path
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import numpy as np
from mmtrack.apis import inference_mot, init_model
from cv2 import imwrite
import os, glob, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2

class DataHandler:

    def __init__(self, 
                 input_folder_path:str,
                 output_video_path:str,
                 config,
                 checkpoint,
                 device,
                 ):

        self.input_path = input_folder_path
        self.output_path = output_video_path
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.crop_folder_path = self.output_path + 'crops/'
        self.cluster_folder_path = self.output_path + 'clusters/'
        self.cam_id = 2

    def track_persons_and_crop_from_dir(self, capture_index:int=500, acc_threshold:int=0.999) -> None:
        """
        capture_index = frame will be captured every capture_index value. input 1 for all frames
        """
        def create_crops_folder() -> None:
            """Creates a crops folder for the given output path"""
            assert self.output_path
            if not osp.isdir(self.crop_folder_path):
                os.mkdir(self.crop_folder_path)

        create_crops_folder()
        if osp.isdir(self.input_path):
            vids = os.listdir(self.input_path)
            print(f'Found the following videos to process: {vids}')
            for vid in vids:
                cur_path = os.path.join(self.input_path, vid)
                print(f'Processing video: {cur_path}')
                if not osp.isdir(cur_path) and '.avi' in cur_path:
                    imgs = mmcv.VideoReader(cur_path)
                    print('begin extraction of video {}'.format(vid))
                    model = init_model(self.config, self.checkpoint, device=self.device)
                    prog_bar = mmcv.ProgressBar(len(imgs))
                    # test and show/save the images
                    for i, img in enumerate(imgs):
                        if isinstance(img, str):
                            img = osp.join(self.input_path, img)
                        prog_bar.update()
                        cap = 0
                        if i % capture_index == 0:
                            cap = 10
                        if cap > 0:  # we capture 10 frames every 500 frames
                            cap -= 1
                            result = inference_mot(model, img, frame_id=i) # using mmtrack inference
                            acc = result['track_results'][0][:, -1]
                            ids = result['track_results'][0][:, 0]
                            mask = np.where(acc > acc_threshold) # accepting accuracy above threshold
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
                                imwrite(cropped_out, per_crop)
                            prog_bar.update()
                        else:
                            prog_bar.update()

    def create_clusters(self, k):
        # we can extend this later to receive input from args
        # if generating the crops in the same run, the input for the clustering should be the output arg
        if not osp.isdir(self.cluster_folder_path):
            os.mkdir(self.cluster_folder_path)
        print(self.cluster_folder_path)
        images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(self.crop_folder_path + '*')]
        paths = [file for file in glob.glob(self.crop_folder_path + '*')]
        images = np.array(np.float32(images).reshape(len(images), -1) / 255)
        model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        predictions = model.predict(images.reshape(-1, 224, 224, 3))
        pred_images = predictions.reshape(images.shape[0], -1)
        kmodel = KMeans(n_clusters=k, random_state=728)
        kmodel.fit(pred_images)
        kpredictions = kmodel.predict(pred_images)
        for i in range(k):
            if not osp.isdir(f'{self.cluster_folder_path}/cluster{str(i)}'):
                os.makedirs(f'{self.cluster_folder_path}/cluster{str(i)}')
        for i in range(len(paths)):
            shutil.copy2(paths[i], f'{self.cluster_folder_path}/cluster{str(kpredictions[i])}')



if __name__ == '__main__':
    pass
    # print(os.curdir)
    # input_path = '/home/bar_cohen/data_samples/1.8.21cam1/'
    # output_path = '/home/bar_cohen/data_samples/1.8.21cam1/output/'
    # config = "/home/bar_cohen/mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py"
    # checkpoint = None # download checkpoint
    # device = None
    # data_handler = DataHandler(input_path, output_path, config, checkpoint,device)
    # print('running tracking and crop')
    # # data_handler.track_persons_and_crop_from_dir(10000)
    # print('running clustering')
    # data_handler.create_clusters(5)
    # print("Done!")