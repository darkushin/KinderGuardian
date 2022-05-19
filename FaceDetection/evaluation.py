from collections import defaultdict
import os
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DataProcessing.dataProcessingConstants import NAME_TO_ID, ID_TO_NAME
from FaceDetection.data_handler import load_data
from FaceDetection.faceClassifer import FaceClassifer

HIST_SAVE_PATH = "/mnt/raid1/home/bar_cohen/FaceData/runs/plots/data_hists/"

def test_accuracy_of_dataset(fc:FaceClassifer, dl:DataLoader, title:str):
    """ Simple accuracy tester for face Classfication """
    print('Accuracy of dataset --- ', title)
    fc.model_ft.eval() # eval mode
    correct = 0
    total = 0
    for batch in dl:
        images, labels = batch
        inputs = images.to(fc.device)
        labels = labels.to(fc.device)
        preds, _ = fc.predict(inputs)
        correct += torch.sum(preds == labels)
        total += len(labels)
    print(correct/ total)


def per_id_test_acc(fc:FaceClassifer,le, exp_name:str, dl:DataLoader,title:str):
    acc_dict = {k:0 for k in NAME_TO_ID.keys()}
    acc_totals = {k: 0 for k in NAME_TO_ID.keys()}
    for batch in dl:
        images, labels = batch
        inputs = images.to(fc.device)
        labels = labels.to(fc.device)
        preds, _  = fc.predict(inputs)
        for p, l in zip(preds, labels):
            if p == l:
                acc_dict[ID_TO_NAME[le.inverse_transform([int(l)])[0]]] += 1
            acc_totals[ID_TO_NAME[le.inverse_transform([int(l)])[0]]] += 1


    for k, v in acc_dict.items():
        if acc_totals[k] > 0:
            acc_dict[k] = acc_dict[k] / acc_totals[k]
    plt.clf()
    df =pd.DataFrame(acc_dict, columns=sorted(NAME_TO_ID.keys()), index=[0])
    ax = sns.barplot(data=df)
    ax.set_xticklabels(df.columns, rotation=45)
    cur_title = f'Acc per ID in {title}'
    plt.title(cur_title)
    plt.savefig(os.path.join(HIST_SAVE_PATH, f'{exp_name} {cur_title}.jpg'))

def build_samples_hist(le, y, exp_name, title:str=None):
    """Create a Bar plot that represented the number of face samples from every id"""
    plt.clf()
    samples_dict = defaultdict(int)
    for id in y:
        samples_dict[ID_TO_NAME[le.inverse_transform([int(id)])[0]]] += 1
    df = pd.DataFrame(samples_dict, columns=sorted(samples_dict.keys()),index=[0])
    ax= sns.barplot(data=df, order=list(df.columns))
    ax.set_xticklabels(df.columns, rotation=45)
    cur_title = f'samples hist of {title} set'
    plt.title(cur_title)
    plt.savefig(os.path.join(HIST_SAVE_PATH, f'{exp_name} {cur_title}.jpg'))
    # plt.show()

def eval_faceClassifier(exp_name, checkpoint_path):
    def run_test_for_dataset(dl, datasplit):
        build_samples_hist(le, dl.dataset.labels , exp_name, datasplit)
        per_id_test_acc(fc, le, exp_name, dl, datasplit)
        test_accuracy_of_dataset(fc, dl, datasplit)


    data_path = '/mnt/raid1/home/bar_cohen/FaceData/'
    num_classes = 21  # num of unique classes
    le, dl_train, dl_val, dl_test = load_data(data_path)
    fc = FaceClassifer(num_classes, le)
    fc.model_ft.load_state_dict(torch.load(checkpoint_path))
    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    run_test_for_dataset(dl_train,'Train')
    run_test_for_dataset(dl_val, 'Val')
    run_test_for_dataset(dl_test, 'Test')