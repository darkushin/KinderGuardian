from argparse import ArgumentParser
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
    parser.add_argument('--reid_checkpoint', help='checkpoint file for reid model')
    parser.add_argument('--device', help='device to run on')
    parser.add_argument('--k_cluster', help='If running clustering on Data Processing')
    parser.add_argument('--capture_index', help='If running track and crop on Data Processing')
    parser.add_argument('--acc_threshold', default=0.8,
                        help='If running track and crop on Data Processing, or if running track-and-reid model')
    parser.add_argument('--model_weights', help='The weights that should be used for inference (reID model)')
    parser.add_argument('--dataset', help='The name of the dataset that should be used in the reID model')
    return parser.parse_args()


def create_optional_args() -> List:
    """
    Creates a list of optional arguments that if given should be appended to the sys.call arguments.
    """
    optional_args: List = []
    if args.model_weights:
        optional_args.extend(['MODEL.WEIGHTS', args.model_weights])

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
    re-id-train --reid_config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml --model_weights
    ./fast-reid/checkpoints/duke_bot_R101-ibn.pth --dataset "DukeMTMC-reID-test"

    Using fast-reid directly:
    config ./fast-reid/configs/DukeMTMC/bagtricks_R101-ibn.yml MODEL.WEIGHTS
    ./fast-reid/checkpoints/duke_bot_R101-ibn.pth MODEL.DEVICE "cuda:0" DATASETS.DATASET "DukeMTMC-reID-test"
    """
    optional_args: List = create_optional_args()
    if args.action == RE_ID_TRAIN:
        script_args = ['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './fast-reid/tools/train_net.py',
                       '--config-file', args.reid_config, 'MODEL.DEVICE', 'cuda:0', 'DATASETS.DATASET', args.dataset]
        script_args.extend(optional_args)
        call(script_args)

    if args.action == RE_ID_EVAL:
        script_args = ['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './fast-reid/tools/train_net.py',
                       '--config-file', args.reid_config, '--eval-only', 'MODEL.DEVICE', 'cuda:0', 'DATASETS.DATASET',
                       args.dataset]
        script_args.extend(optional_args)
        call(script_args)


def validate_tracking_args():
    # TODO transfer the validation to each sub-module
    assert args.config is not None, 'tracking action requires a `--config` argument'
    assert args.input is not None, 'tracking action requires a `--input` argument'
    assert args.output is not None, 'tracking action requires a `--output` argument'


def execute_combined_model():
    call(['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', './models/track_and_reid_model.py',
          args.track_config, args.reid_config, '--input', args.input, '--output', args.output, '--acc_th', args.acc_threshold,
          '--reid_opts', 'DATASETS.DATASET', args.dataset])



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
