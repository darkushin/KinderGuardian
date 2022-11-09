import os
import glob
import math
import logging
import numpy as np
import os.path as osp

CLOTHES_LABEL = 1
CAM_ID = 1


class Street42(object):
    """ CCVID

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.
    """

    def __init__(self, root='/data/datasets/', seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root)
        self.gallery_path = self.root
        self._check_before_run()

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(self.gallery_path)

        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len,
                                                                                     stride=stride)

        num_imgs_per_tracklet = num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_gallery_pids
        num_total_clothes = 0
        num_total_tracklets = num_gallery_tracklets

        logger = logging.getLogger('reid.dataset')
        logger.info("=> CCVID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------")
        logger.info("  subset       | # ids | # tracklets | # clothes")
        logger.info("  ---------------------------------------------")
        logger.info("  gallery      | {:5d} | {:11d} ".format(num_gallery_pids, num_gallery_tracklets))
        logger.info("  ---------------------------------------------")
        logger.info(
            "  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        logger.info("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        logger.info("  ---------------------------------------------")

        self.gallery = gallery

        self.recombined_gallery = recombined_gallery
        self.gallery_vid2clip_index = gallery_vid2clip_index

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _process_data(self, data_path):
        tracklet_path_list = []
        pid_container = set()

        for tracklets_path in glob.glob(data_path + '*/*/*', recursive=True):
            pid = self.get_pid_from_path(tracklets_path)
            pid_container.add(pid)
            tracklet_path_list.append((tracklets_path, pid, CLOTHES_LABEL))

        pid_container = sorted(pid_container)
        num_pids = len(pid_container)

        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*'))
            img_paths.sort()
            pid = int(pid)
            camid = CAM_ID
            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, CLOTHES_LABEL))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def get_pid_from_path(self, img_path):
        """
        Given a path to the gallery image, return its pid.
        Example: '/home/bar_cohen/raid/42street/labeled_track_gallery_o/0008/v037_t0000003', return '0008'
        """
        return img_path.split(os.sep)[-2]

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx: end_idx: stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx: end_idx: new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()

