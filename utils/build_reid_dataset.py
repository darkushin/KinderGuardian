import yaml
from argparse import ArgumentParser
import os
import sys
import shutil
from distutils.dir_util import copy_tree

"""
This script should be used to build datasets using different configurations and combinations of data days
"""

# CONSTANTS:
TRAIN_SET = 'bounding_box_train'
TEST_SET = 'bounding_box_test'
QUERY_SET = 'query'


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--override', action='store_true', help='override an existing dataset with the same name')
    args = parser.parse_args()
    return args


def create_dataset_folder():
    """
    Create a new dataset folder according to the given dataset name and output path.
    Displays an error in case a dataset with the same name already exists.
    """
    dataset_name = dataset_conf.get('NAME')
    dataset_path = os.path.join(dataset_conf.get('LOCATION'), dataset_name)
    if os.path.exists(dataset_path):
        if not args.override:
            print(f'WARNING! Dataset {dataset_name} already exists in folder {dataset_conf.get("LOCATION")}. '
                  f'If you want to override it, use the script argument `--override`')
            sys.exit(1)
        else:
            shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)


def create_readme():
    """
    Creates a README file in the dataset folder which explains the configurations of this dataset
    """
    readme = open(os.path.join(dataset_conf.get("LOCATION"), dataset_conf.get("NAME"), "README.md"), "w+")
    readme.write(f'**Description**: {dataset_conf.get("DESCRIPTION")}\n\n')
    readme.write(f'The dataset contains the following data-days: \n'
                 f'- **Train**: {dataset_conf.get("TRAIN")}\n'
                 f'- **Test**: {dataset_conf.get("TEST")}\n'
                 f'- **Query**: {dataset_conf.get("QUERY")}\n')
    readme.close()


def copy_imgs():
    """
    Copies all the images from the specified Train/Test/Query folders to the corresponding folder in the dataset
    """
    dataset_name = dataset_conf.get('NAME')
    dataset_path = os.path.join(dataset_conf.get('LOCATION'), dataset_name)
    train_folder = os.path.join(dataset_path, TRAIN_SET)
    test_folder = os.path.join(dataset_path, TEST_SET)
    query_folder = os.path.join(dataset_path, QUERY_SET)
    labeled_data_path = dataset_conf.get('LABELED_DATA')

    if dataset_conf.get('TRAIN'):
        train_days = dataset_conf.get('TRAIN').split(',')
        for train_day in train_days:
            copy_tree(os.path.join(labeled_data_path, train_day), train_folder)

    if dataset_conf.get('TEST'):
        test_days = dataset_conf.get('TEST').split(',')
        for test_day in test_days:
            copy_tree(os.path.join(labeled_data_path, test_day), test_folder)

    if dataset_conf.get('QUERY'):
        query_days = dataset_conf.get('QUERY').split(',')
        for query_day in query_days:
            copy_tree(os.path.join(labeled_data_path, query_day), query_folder)


if __name__ == '__main__':
    import os
    # os.makedirs("/mnt/raid1/home/bar_cohen/42street/part3_all/", exist_ok=False)
    # os.makedirs("/mnt/raid1/home/bar_cohen/42street/part3_all/bounding_box_test", exist_ok=False)
    # os.makedirs("/mnt/raid1/home/bar_cohen/42street/part3_all/bounding_box_train", exist_ok=False)
    # os.makedirs("/mnt/raid1/home/bar_cohen/42street/part3_all/query", exist_ok=False)
    global_i = 0
    for i, folder in enumerate(os.listdir("/mnt/raid1/home/bar_cohen/42street/part3_galleries/")):
        full_path = os.path.join("/mnt/raid1/home/bar_cohen/42street/part3_galleries/", folder)
        for root, _, files in os.walk(full_path):
            for file in files:
                label = file[0:4]
                new_pic_name = f'{label}_c{i}_f{global_i:07d}.jpg'
                cur_file  = os.path.join(root,file)
                shutil.copy(cur_file, os.path.join(file, f"/mnt/raid1/home/bar_cohen/42street/part3_all/bounding_box_test/{new_pic_name}"))
                global_i += 1

    # args = get_args()
    # with open("utils/dataset_configs.yml", "r") as stream:
    #     dataset_conf = yaml.safe_load(stream)
    #
    # create_dataset_folder()
    # copy_imgs()
    # create_readme()
    #
    # print(f'Dataset Created Successfully! Checkout {os.path.join(dataset_conf.get("LOCATION"), dataset_conf.get("NAME"))}'
    #       f' and verify that the README file describes correctly the dataset.')

