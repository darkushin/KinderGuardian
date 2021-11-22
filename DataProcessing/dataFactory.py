import logging

from dataProcessingConstants import *
from dataHandler import DataHandler

def data_factory(action:str, video_folder_input_path:str, output_folder_path:str, config:object, checkpoint:str=None,
                 device:str=None, k_cluster:int=None, capture_index:int=None, acc_threshold:float=None) -> None:
    """
    assumes action, input, output and config are validated by model runner.
    """
    dh = DataHandler(input_folder_path=video_folder_input_path, output_video_path=output_folder_path, config=config,
                     checkpoint=checkpoint, device=device)
    if action == CLUSTER:
        if not k_cluster:
            logging.warning(f"No K-cluster received, running with default k {K_CLUSTERS}")
            k_cluster = K_CLUSTERS
        dh.create_clusters(k_cluster)

    elif action == TRACK_AND_CROP:
        if not capture_index:
            logging.warning(f"No capture-index received, running with default {CROP_INDEX}")
            capture_index = CROP_INDEX
        if not acc_threshold:
            logging.warning(f"No confidence tracking treshold received, running with default {ACC_THRESHOLD}")
            acc_threshold = ACC_THRESHOLD
        logging.info('running Track and Crop')
        dh.track_persons_and_crop_from_dir(capture_index=capture_index, acc_threshold=acc_threshold)
        
    else:
        raise Exception(f'Unsupported action! run model_runner.py -h to see the list of possible actions')
