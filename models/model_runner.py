from argparse import ArgumentParser
from subprocess import call
import sys


def get_args():
    parser = ArgumentParser()
    parser.add_argument('action', help='what action would you like to run?', choices=['tracking'])
    parser.add_argument('--config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument('--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    return parser.parse_args()


def execute_tracking_action():
    """
    Usage example:
    tracking --input /home/bar_cohen/KinderGuardian/mmtracking/demo/demo.mp4 --output /home/bar_cohen/KinderGuardian/test-output.mp4  --config /home/bar_cohen/KinderGuardian/mmtracking/configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py
    """
    call(['/home/bar_cohen/miniconda3/envs/mmtrack/bin/python', '../mmtracking/demo/demo_mot_vis.py', args.config,
          '--input', args.input, '--output', args.output])


def validate_tracking_args():
    assert args.config is not None, 'tracking action requires a `--config` argument'
    assert args.input is not None, 'tracking action requires a `--input` argument'
    assert args.output is not None, 'tracking action requires a `--output` argument'


def runner():
    if args.action == 'tracking':
        validate_tracking_args()
        execute_tracking_action()
    else:
        raise Exception(f'Unsupported action! run model_runner.py -h to see the list of possible actions')


if __name__ == '__main__':
    args = get_args()
    runner()
