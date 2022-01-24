from argparse import ArgumentParser, REMAINDER
from subprocess import call

import sys
from models.model_constants import *
import DataProcessing.dataFactory
from typing import List


def get_args():
    parser = ArgumentParser()
    parser.add_argument('action', help='what action would you like to run?', choices=CHOICES)
    parser.add_argument('--track_config', help='config file for mmtracking model')
    parser.add_argument('--reid_config', help='config file for reid model')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument('--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--mmtrack_checkpoint', help='checkpoint file for mmtrack model')
    parser.add_argument('--device', default='cuda:0', help='device to run on')
    parser.add_argument('--k_cluster', help='If running clustering on Data Processing')
    parser.add_argument('--capture_index', help='If running track and crop on Data Processing')
    parser.add_argument('--acc_threshold', default=0.8,
                        help='If running track and crop on Data Processing, or if running track-and-reid model')
    parser.add_argument('--reid_opts', help='Modify config options using the command-line', default=None, nargs=REMAINDER)
    parser.add_argument('--crops_folder')

    return parser.parse_args()


def create_reid_opts() -> List:
    """
    Creates a list of optional arguments that should be passed to the reid model through the `--reid_opts` param.
    """
    reid_opts: List = []
    if args.reid_opts:
        for opt in args.reid_opts:
            reid_opts.append(opt)
    reid_opts.extend(['MODEL.DEVICE', args.device])
    return reid_opts


def create_optional_args() -> List:
    """
    Creates a list of optional arguments that should be passed to the tracking model.
    """
    optional_args: List = []
    if args.mmtrack_checkpoint:
        optional_args.extend(['--track_checkpoint', args.mmtrack_checkpoint])
    if args.device:
        optional_args.extend(['--device', args.device])
    if args.acc_threshold:
        optional_args.extend(['--acc_th', args.acc_threshold])
    if args.crops_folder:
        optional_args.extend(['--crops_folder', args.crops_folder])
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
                       '--config-file', args.reid_config, '--eval-only', 'MODEL.DEVICE', 'cuda:1', 'DATASETS.DATASET',
                       args.dataset]
        script_args.extend(reid_opts)
        call(script_args)


def validate_tracking_args():
    # TODO transfer the validation to each sub-module
    assert args.config is not None, 'tracking action requires a `--config` argument'
    assert args.input is not None, 'tracking action requires a `--input` argument'
    assert args.output is not None, 'tracking action requires a `--output` argument'


def execute_combined_model():
    """
    Usage example:
    re-id-and-tracking --track_config ./mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py
    --reid_config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml --input ../Data-Shoham/1.8.21_cam1/videos/IPCamera_20210801095724.avi
    --output ./Results/Reid-Eval2-2.8-test.mp4 --acc_th 0.98 --crops_folder /mnt/raid1/home/bar_cohen/DB_Crops/
    --reid_opts DATASETS.DATASET inference_on_train_data MODEL.WEIGHTS ./fast-reid/checkpoints/scratch-id-by-day.pth

    """
    reid_opts: List = create_reid_opts()
    optional_args: List = create_optional_args()
    script_args = ['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './models/track_and_reid_model.py',
          args.track_config, args.reid_config, '--input', args.input, '--output', args.output]
    script_args.extend(optional_args)
    script_args.append('--reid_opts')
    script_args.extend(reid_opts)
    print(script_args)
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
    args = get_args()
    runner()
