#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')
sys.path.append('fast-reid')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


'''
USAGE:
need to be under the /fast-reid folder and then run:
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R101-ibn.yml MODEL.WEIGHTS 
checkpoints/duke_bot_R101-ibn-new.pth MODEL.DEVICE "cuda:0" DATASETS.DATASET "DukeMTMC"
'''


def validate_args(args):
    assert args.config_file, 'No config file given, see `/fast-reid/tools/train_net.py` for usage example of ReID model'
    # assert args.DATASETS.DATASET, 'Dataset name must be given, see `/fast-reid/tools/train_net.py` for usage example of ReID model'


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    validate_args(args)
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
