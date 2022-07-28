import os
import warnings
from argparse import ArgumentParser, REMAINDER
from subprocess import call

import sys

from DataProcessing.DB.dal import Crop, get_entries
from DataProcessing.utils import viz_DB_data_on_video
from models.model_constants import *
import DataProcessing.dataFactory
from typing import List


def get_args():
    parser = ArgumentParser()
    parser.add_argument('action', help='what action would you like to run?', choices=CHOICES)
    parser.add_argument('--track_config', help='config file for mmtracking model')
    parser.add_argument('--reid_config', help='config file for reid model')
    parser.add_argument('--pose_config', help='config file for pose estimation model')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument('--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--mmtrack_checkpoint', help='checkpoint file for mmtrack model')
    parser.add_argument('--pose_checkpoint', help='checkpoint file for pose estimation model')
    parser.add_argument('--device', default='cuda:0', help='device to run on')
    parser.add_argument('--k_cluster', help='If running clustering on Data Processing')
    parser.add_argument('--capture_index', help='If running track and crop on Data Processing')
    parser.add_argument('--acc_threshold', default=0.8,
                        help='If running track and crop on Data Processing, or if running track-and-reid model')
    parser.add_argument('--reid_opts', help='Modify config options using the command-line', default=None, nargs=REMAINDER)
    parser.add_argument('--crops_folder')
    parser.add_argument('--inference_only', action='store_true', help='use the tracking and reid model for inference')
    parser.add_argument('--db_tracklets', action='store_true', help='use the tagged DB to create tracklets for inference')
    parser.add_argument('--experiment_mode', action='store_true', help='run in experiment_mode')
    parser.add_argument('--exp_description', help='The description of the experiment that should appear in the ablation study output')
    parser.add_argument('--reid_model', choices=['fastreid', 'ctl'], default='fastreid', help='Reid model that should be used.')

    return parser.parse_args()


def create_reid_opts() -> List:
    """
    Creates a list of optional arguments that should be passed to the reid model through the `--reid_opts` param.
    """
    reid_opts: List = []
    if args.reid_opts:
        for opt in args.reid_opts:
            reid_opts.append(opt)
    if args.device and args.reid_model == 'fastreid':
        reid_opts.extend(['MODEL.DEVICE', args.device])
    return reid_opts


def create_optional_args() -> List:
    """
    Creates a list of optional arguments that should be passed to the tracking model.
    """
    optional_args: List = []
    if args.mmtrack_checkpoint:
        optional_args.extend(['--track_checkpoint', args.mmtrack_checkpoint])
    if args.pose_config:
        optional_args.extend(['--pose_config', args.pose_config])
    if args.pose_checkpoint:
        optional_args.extend(['--pose_checkpoint', args.pose_checkpoint])
    if args.device:
        optional_args.extend(['--device', args.device])
    if args.acc_threshold:
        optional_args.extend(['--acc_th', args.acc_threshold])
    if args.crops_folder:
        optional_args.extend(['--crops_folder', args.crops_folder])
    if args.inference_only:
        optional_args.extend(['--inference_only'])
    if args.db_tracklets:
        optional_args.extend(['--db_tracklets'])
    if args.experiment_mode:
        optional_args.extend(['--experiment_mode'])
    if args.exp_description:
        optional_args.extend(['--exp_description', args.exp_description])
    if args.reid_model:
        optional_args.extend(['--reid_model', args.reid_model])
    return optional_args


def execute_tracking_action():
    """
    Usage example:
    tracking --input /home/bar_cohen/KinderGuardian/mmtracking/demo/demo.mp4 --output
    /home/bar_cohen/KinderGuardian/test-output.mp4  --track_config
    /home/bar_cohen/KinderGuardian/mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py
    """
    call(['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './mmtracking/demo/demo_mot_vis.py', args.mmtrack_config,
          '--input', args.input, '--output', args.output])


def execute_reid_action():
    """
    Using model_runner:
    re-id-train --reid_config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml --device cuda:1 --reid_opts
    MODEL.WEIGHTS ./fast-reid/checkpoints/duke_bot_R101-ibn.pth DATASETS.DATASET query-2.8_test-4.8

    Using fast-reid directly:
    config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml MODEL.WEIGHTS
    ./fast-reid/checkpoints/duke_bot_R101-ibn.pth MODEL.DEVICE "cuda:0" DATASETS.DATASET "DukeMTMC-reID-test"
    """
    reid_opts: List = create_reid_opts()
    if args.action == RE_ID_TRAIN:
        script_args = ['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './fast-reid/tools/train_net.py',
                       '--config-file', args.reid_config, '--machine-rank', '1']
        script_args.extend(reid_opts)
        call(script_args)

    if args.action == RE_ID_EVAL:
        script_args = ['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './fast-reid/tools/train_net.py',
                       '--config-file', args.reid_config, '--eval-only']
        script_args.extend(reid_opts)
        call(script_args)


def validate_tracking_args():
    # TODO transfer the validation to each sub-module
    assert args.config is not None, 'tracking action requires a `--config` argument'
    assert args.input is not None, 'tracking action requires a `--input` argument'
    assert args.output is not None, 'tracking action requires a `--output` argument'

def get_query_set():
    query_set = []
    for _ , _ , files in os.walk("/mnt/raid1/home/bar_cohen/trimmed_videos/"):
        for file in files:
            is_tagged = len(get_entries(filters={Crop.vid_name == file[9:-4], Crop.reviewed_one == True}).all()) > 0
            if file[13:17] in ['0808','0730','0804']  and is_tagged:
                query_set.append(file)
    return query_set


def execute_combined_model():
    """
    Usage example:

    fastreid configs:
    re-id-and-tracking
    --track_config
    ./mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py
    --mmtrack_checkpoint
    /home/bar_cohen/KinderGuardian/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
    --reid_config
    ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml
    --input
    /mnt/raid1/home/bar_cohen/trimmed_videos/IPCamera_20210803105422/IPCamera_20210803105422_s0_e501.mp4
    --output
    /mnt/raid1/home/bar_cohen/labled_videos/20210803105422_s0_e501_new_model.mp4
    --acc_th
    0.8
    --crops_folder
    /mnt/raid1/home/bar_cohen/DB_Test/
    --inference_only
    --db_tracklets
    --exp_description
    "Compare reid models - fastreid - same_day_0808"
    --device
    cuda:0
    --experiment_mode
    --reid_model
    fastreid
    --reid_opts
    DATASETS.DATASET
    /home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0808/
    MODEL.WEIGHTS
    /home/bar_cohen/KinderGuardian/fast-reid/checkpoints/duke_bot_R101-ibn.pth
    DATALOADER.NUM_WORKERS 0


    CTL configs:
    re-id-and-tracking
    --track_config
    ./mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py
    --mmtrack_checkpoint
    /home/bar_cohen/KinderGuardian/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
    --reid_config
    ./centroids_reid/configs/256_resnet50.yml
    --input
    /mnt/raid1/home/bar_cohen/trimmed_videos/IPCamera_20210803105422/IPCamera_20210803105422_s0_e501.mp4
    --output
    /mnt/raid1/home/bar_cohen/labled_videos/20210803105422_s0_e501_new_model.mp4
    --acc_th
    0.8
    --crops_folder
    /mnt/raid1/home/bar_cohen/DB_Test/
    --inference_only
    --db_tracklets
    --exp_description
    "Pose estimation debugging - ignore this experiment"
    --device
    cuda:0
    --experiment_mode
    --reid_model
    ctl
    --pose_config
    /home/bar_cohen/D-KinderGuardian/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py
    --pose_checkpoint
    /home/bar_cohen/D-KinderGuardian/checkpoints/mmpose-hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
    --reid_opts
    TEST.IMS_PER_BATCH
    128
    TEST.ONLY_TEST
    True
    TEST.WEIGHT
    /home/bar_cohen/D-KinderGuardian/centroids_reid/checkpoints/query_0730_0808_dataset_epoch79.ckpt
    DATASETS.NAMES
    dukemtmcreid
    DATASETS.ROOT_DIR
    /home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0808_verified/bounding_box_test
    MODEL.USE_CENTROIDS
    True
    GPU_IDS
    [1]
    DATALOADER.NUM_WORKERS
    0

    ByteTracker:
    re-id-and-tracking --track_config ./mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py
    --mmtrack_checkpoint /home/bar_cohen/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
    --reid_config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml --input /home/bar_cohen/KinderGuardian/Videos/trimmed_1.8.21-095724.mp4
    --output ./Results/trimmed-bytetrack_labeled.mp4 --acc_th 0.7 --crops_folder /mnt/raid1/home/bar_cohen/DB_Test/
    --device cuda:1 --reid_opts DATASETS.DATASET inference_on_train_data MODEL.WEIGHTS ./fast-reid/checkpoints/1.8.21-model.pth
    """
    reid_opts: List = create_reid_opts()
    optional_args: List = create_optional_args()
    inference_output = "/mnt/raid1/home/bar_cohen/labled_videos/inference_videos"
    # print('Total videos in eval set:', len(get_query_set()))
    street42 = ["/mnt/raid1/home/bar_cohen/42street/42street_tagged_vids/part3/" , "/mnt/raid1/home/bar_cohen/42street/42street_tagged_vids/part2/", "/mnt/raid1/home/bar_cohen/42street/42street_tagged_vids/part1/"]
    query_set = [os.path.join(part, vid) for part in street42 for vid in os.listdir(part)]
    for query_vid in query_set:
        # if '20210808' not in query_vid:
        #     print(f'skipping {query_vid}')
        #     continue
        print(f'running {query_vid}')
        args.input = os.path.join('/mnt/raid1/home/bar_cohen/trimmed_videos',
                                  query_vid.split('_')[0]+'_'+query_vid.split('_')[1],
                                  query_vid)

        args.output = os.path.join(inference_output, 'inference_' + query_vid.split('/')[-1])
        script_args = ['/home/bar_cohen/miniconda3/envs/CTL/bin/python3.7', './models/track_and_reid_model.py',
                       args.track_config, args.reid_config, '--input', args.input, '--output', args.output]

        script_args.extend(optional_args)
        script_args.append('--reid_opts')
        script_args.extend(reid_opts)
        call(script_args)

def runner():
    if args.action == TRACKING:
        # validate_tracking_args()
        execute_tracking_action()

    elif args.action in DATA_PROCESSING_ACTIONS:
        validate_tracking_args()
        DataProcessing.dataFactory.data_factory(action=args.action, video_folder_input_path=args.input,
                                                output_folder_path=args.output, config=args.config,
                                                checkpoint=args.checkpoint,
                                                k_cluster=args.k_cluster,
                                                acc_threshold=args.acc_threshold,
                                                capture_index=args.capture_index)

    elif args.action in REID_ACTIONS:
        execute_reid_action()

    elif args.action == RE_ID_AND_TRACKING:
        execute_combined_model()
    else:
        raise Exception(f'Unsupported action! run model_runner.py -h to see the list of possible actions')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = get_args()
    runner()
