import logging
import os
from DataProcessing.dataProcessingConstants import *
from DataProcessing.dataHandler import DataHandler

def assert_input_args(video_folder_input_path: str, output_folder_path: str, config: object) -> None:
    """
    Function that asserts the required args were entered correctly.
    """
    assert os.path.isdir(video_folder_input_path) , "Input must be a non-empty folder with videos"
    assert os.path.isdir(output_folder_path) , "Output must be a folder"
    assert config is not None, "A path to a config file for mmtracking must be entered"


def data_factory(action: str, video_folder_input_path: str, output_folder_path: str, config: object,
                 checkpoint: str = None,
                 device: str = None, k_cluster: int = None, capture_index: int = None,
                 acc_threshold: float = None) -> None:
    """
    Data Factory to parse actions coming from script runner. the none-default args are expected to be validated.
    @param action: Action to preform using DataHandler
    @param input_folder_path: Input folder for videos to parse. must be non-empty with avi format videos
    @param output_video_path: Output Folder for crops and clusters
    @param config: the config file path for the mm-track module
    @param checkpoint: the checkpoint file path for the mm-track module, if non entered downloads checkpoint during runtime.
    @param device: the device to run script from. Defaults to GPU if one exists
    @param k_cluster: Number of clusters to create using clustering.
    @param capture_index: a frame will be captured and analyzed was capture_index frames have passed
    @param acc_threshold: a crop will be saved only if the model's confidence is above acc_treshold
    @return: None
    """
    assert_input_args(video_folder_input_path, output_folder_path, config)
    dh = DataHandler(input_folder_path=video_folder_input_path, output_video_path=output_folder_path, config=config,
                     checkpoint=checkpoint, device=device)
    if action == CLUSTER:
        if not k_cluster:
            logging.warning(f"No K-cluster received, running with default k {K_CLUSTERS}")
            k_cluster = K_CLUSTERS
        dh.create_clusters(k_cluster)
        logging.info(f'Done creating clusters from crops at input dir, please find output in {output_folder_path}')


    elif action == TRACK_AND_CROP:
        if not capture_index:
            logging.warning(f"No capture-index received, running with default {CROP_INDEX}")
            capture_index = CROP_INDEX
        if not acc_threshold:
            logging.warning(f"No confidence tracking treshold received, running with default {ACC_THRESHOLD}")
            acc_threshold = ACC_THRESHOLD
        logging.info('running Track and Crop')
        dh.track_persons_and_crop_from_dir(capture_index=capture_index, acc_threshold=acc_threshold)
        logging.info(f'Done creating crops from videos at input dir, please find output in {output_folder_path}')

    else:
        raise Exception(f'Unsupported action! run model_runner.py -h to see the list of possible actions')
