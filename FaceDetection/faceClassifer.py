import os
import shutil
import sys

import cv2
import mmcv
sys.path.append('DataProcessing')

from DataProcessing.DB.dal import get_entries, Crop
from FaceDetection.faceDetector import FaceDetector


import pandas as pd
import copy
import pickle
import time
from collections import defaultdict
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms, utils, datasets, models
from torchvision.utils import make_grid

from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.facenet_pytorch import InceptionResnetV1, training
from torch import nn, optim
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# helper classes for faceClassifer class
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

#
class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, images:[], labels:[]):
        super(FacesDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        image = self.images[item]
        return image, label


class FaceClassifer():
    """
    The face Classifier is a CNN that utilizes InceptionResnetV1 architecture with a pretrain
    weights of vggface2 to fine-tune custom data
    """
    def __init__(self, num_classes, label_encoder, device='cuda:0'):
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        #print(self.model_ft)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(),lr=0.0001)
        self.writer = SummaryWriter('/mnt/raid1/home/bar_cohen/FaceData/runs/images/')
        self.writer.iteration = 0
        self.writer.interval = 10
        self.softmax = nn.Softmax(dim=1)
        self.le = label_encoder
        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ft, [5,10])


    def _create_Incepction_Resnet_for_finetunning(self):

        # only adding last linear layer, Softmax is added on predict
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=True)
        # for param in model_ft.parameters():
        #     param.requires_grad = False # we dont want to train the original layers
        model_ft.logits = torch.nn.Linear(512, self.num_classes)
        model_ft = model_ft.to(self.device)



        # another option - freeze and add 5 layers, works less well
        # layer_list = list(model_ft.children())[-5:] # final layers
        # model_ft = torch.nn.Sequential(*list(model_ft.children())[:-5]) # skipping 5 last layers
        # # for param in model_ft.parameters():
        # #     param.requires_grad = False # we dont want to train the original layers
        #
        # model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        # model_ft.last_linear = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(in_features=1792, out_features=512, bias=False),
        #     normalize())
        # model_ft.logits = nn.Linear(layer_list[4].in_features, self.num_classes)
        # model_ft.softmax = nn.Softmax(dim=1)
        # model_ft = model_ft.to(self.device)
        return model_ft

    def write_images_to_tensorboard(self,dl_iter:iter, epoch:int):
        pass

    def run_training(self,checkpoint_path, dl_train:DataLoader, dl_val:DataLoader,model_name:str, num_epochs=25) -> None:
        self.model_ft.eval()
        dataloader_iter_train = iter(dl_train)
        dataloader_iter_val = iter(dl_val)

        images , target = next(dataloader_iter_val)
        img_grid = make_grid(images, normalize=False)
        self.writer.add_image('val_image_inputs', img_grid,global_step=0)

        img_grid = make_grid(images, normalize=True)
        self.writer.add_image('val_image_inputs', img_grid,global_step=1)
        metrics = {'acc': training.accuracy}
        with torch.no_grad():
            eval_loss, eval_metric = training.pass_epoch(self.model_ft,
                                self.criterion,
                                dl_val, batch_metrics=metrics,
                                show_running=True,
                                device=self.device,
                                writer=self.writer)

        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            images, target = next(dataloader_iter_train)
            img_grid = make_grid(images, normalize=True)
            self.writer.add_image('train_image_input', img_grid,global_step=epoch+2)
            self.model_ft.train()
            training.pass_epoch(
                self.model_ft, self.criterion, dl_train, self.optimizer_ft,
                self.exp_lr_scheduler,
                batch_metrics=metrics, show_running=True, device=self.device,
                writer=self.writer
            )
            with torch.no_grad():
                images, target = next(dataloader_iter_val)
                img_grid = make_grid(images, normalize=True)
                self.writer.add_image('val_image_inputs', img_grid,global_step=epoch+2)
                self.model_ft.eval()
                eval_loss, eval_metric = training.pass_epoch(self.model_ft,
                                self.criterion,
                                dl_val, batch_metrics=metrics,
                                show_running=True,
                                device=self.device,
                                writer=self.writer)

            best_model_wts = copy.deepcopy(self.model_ft.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_path ,f"{model_name}, {epoch + 1}.pth"))

    def create_data_loaders(self, X_train ,y_train, X_val ,y_val, X_test, y_test, data_path) -> (DataLoader, DataLoader, DataLoader):
        dl_train = DataLoader(FacesDataset(X_train, y_train), batch_size=100, shuffle=True)
        dl_val = DataLoader(FacesDataset(X_val,y_val), batch_size=10, shuffle=True)
        dl_test = DataLoader(FacesDataset(X_test,y_test), batch_size=10, shuffle=True)
        # must save datasets dls for reproducibility
        pickle.dump(dl_train, open(os.path.join(data_path, 'dl_train_1.pkl'), 'wb'))
        pickle.dump(dl_val, open(os.path.join(data_path, 'df_val_1.pkl'), 'wb'))
        pickle.dump(dl_test, open(os.path.join(data_path, 'df_test_1.pkl'), 'wb'))
        return dl_train, dl_val, dl_test

    def imshow(self, inp, labels=None):
        """Imshow for Tensor."""
        plt.figure()
        f , ax = plt.subplots(len(inp),1)
        for i, img in enumerate(inp):
            img = img.permute(1,2,0).int().numpy()
            ax[i].imshow(img)
            if labels:
                ax[i].set_ylabel(labels[i])
        plt.show()

    def predict(self, inputs):
        """Given input images as tensors, return predictions"""
        with torch.no_grad():
            outputs = self.softmax(self.model_ft(inputs.to(self.device)))
            preds = torch.argmax(outputs, dim=1)
            return preds, outputs

    def get_accuracy(self, preds, labels):
        """Acc given prediction and true labels"""
        correct = 0
        assert len(preds) == len(labels)
        for p, l in zip(preds, labels):
            if p == l:
                correct += 1
        return correct / (len(preds))

    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def plot_results(self, metrics):
        """ Simple plots of training and validation """
        sns.lineplot(x=range(len(metrics['batch_train_loss'])), y=metrics['batch_train_loss'], color='red')
        plt.xlabel('Batch num')
        plt.ylabel('Train Loss')
        plt.title('Training Batch Loss')
        plt.show()

        sns.lineplot(x=range(len(metrics['batch_val_loss'])), y=metrics['batch_val_loss'], color='orange')
        plt.xlabel('Batch num')
        plt.ylabel('Train Loss')
        plt.title('Validation Batch Loss')
        plt.show()

        acc_df = pd.DataFrame({'val_acc_by_epoch': metrics['val_acc_by_epoch'],
                               'train_acc_by_epoch':metrics['train_acc_by_epoch'],
                               'epoch': range(len(metrics['train_acc_by_epoch']))})

        sns.lineplot(data=acc_df,x='epoch', y='val_acc_by_epoch', color='blue')
        sns.lineplot(data=acc_df,x='epoch', y='train_acc_by_epoch', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy by Epoch')
        plt.legend(['Validation', 'Training'])
        plt.show()


def test_accuracy_of_dataset(fc:FaceClassifer, dl:DataLoader, name:str):
    """ Simple accuracy tester for face Classfication """
    print('Accuracy of dataset --- ', name)
    fc.model_ft.eval() # eval mode
    acc = []
    for batch in dl:
        images, labels = batch
        inputs = images.to(fc.device)
        labels = labels.to(fc.device)
        preds, _ = fc.predict(inputs)
        acc.append(fc.get_accuracy(preds, labels))
    print(np.mean(acc))

def labelencode(label_encoder_output,X_train ,y_train, X_val ,y_val, X_test, y_test ,classes_to_drop:list):
    def drop_from_cur_set(X, y, le):
        indexs_to_drop = [index for index, cls in enumerate(y) if cls in classes_to_drop]
        X = [x for i, x in enumerate(X) if i not in indexs_to_drop]
        y = [cls for i, cls in enumerate(y) if i not in indexs_to_drop]
        y = le.transform(y)
        assert len(X) == len(y)
        return X,y

    """
    creates a label encoder based on inserted classes to drop
    Args:
        label_encoder_output: output path to save label encoder
        X_train ,X_val, X_test: the image tensors
        y_train, y_val, y_test: image labels
        classes_to_drop: label classes not to be trained upon

    Returns: X_train_transformed, y_train_transformed, .. , label-encoder

    """
    # creating an all y vector to account for all images
    y = [int(i) for cur_y_set in [y_train,y_val,y_test] for i in cur_y_set if i not in classes_to_drop]
    print(set(y))
    le = preprocessing.LabelEncoder()
    le.fit(y)
    print(le.classes_)
    X_train ,y_train = drop_from_cur_set(X_train,y_train, le)
    X_val ,y_val = drop_from_cur_set(X_val,y_val, le)
    X_test ,y_test = drop_from_cur_set(X_test,y_test , le)
    pickle.dump(le, open(os.path.join(label_encoder_output, 'le.pkl'), 'wb'))
    return X_train,y_train,X_val,y_val,X_test,y_test, le

def load_data(data_path):
    le = pickle.load(open(os.path.join(data_path, 'le.pkl'), 'rb'))
    dl_train = pickle.load(open(os.path.join(data_path, 'dl_train_1.pkl'), 'rb'))
    dl_val = pickle.load(open(os.path.join(data_path, 'df_val_1.pkl'), 'rb'))
    dl_test = pickle.load(open(os.path.join(data_path, 'df_test_1.pkl'), 'rb'))
    return le, dl_train, dl_val, dl_test

# TODO deprecated use, remove when sure
def create_face_data_from_db() -> None:
    face_crops = get_entries(filters={Crop.is_face == True,
                                      Crop.reviewed_one == True ,
                                      Crop.is_vague == False ,
                                      Crop.invalid == False}).all()
    crops_path = "/mnt/raid1/home/bar_cohen/"
    data_path = '/mnt/raid1/home/bar_cohen/FaceData/images/'
    for crop in face_crops:
        if crop.vid_name[4:8] not in ['0808','0730']:
            shutil.copy(os.path.join(crops_path, crop.vid_name, crop.im_name), os.path.join(data_path, crop.im_name))
    print('done')

def classify_list_of_images(num_classes, le, checkpoint_path):
    fc = FaceClassifer(num_classes=num_classes, label_encoder=le)
    fc.model_ft.load_state_dict(torch.load(checkpoint_path))
    fc.model_ft.eval()  # this will train the model after loading weights

def eval_faceClassifier(checkpoint_path):
    data_path = '/mnt/raid1/home/bar_cohen/FaceData/'
    num_classes = 21  # num of unique classes
    le, dl_train, dl_val, dl_test = load_data(data_path)
    fc = FaceClassifer(num_classes, le)
    fc.model_ft.load_state_dict(torch.load(checkpoint_path))
    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    test_accuracy_of_dataset(fc, dl_train, 'Train')
    test_accuracy_of_dataset(fc, dl_val, 'Val')
    test_accuracy_of_dataset(fc, dl_test, 'Test')


def main_train(data_path:str,run_name:str, reload_images_from_db:bool, recreate_data:bool, checkpoint_path:str, load_checkpoint:str):
    """
    This is an example of a training pipeline. Insert a raw images path or existing
    faces data path to build a dataset and train a faceClassifer on it. Data will be saved
    as well metrics for the training.
    Returns:

    """
    fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97])
    print('loading finding faces base on DB tags, or reading from pickle...')
    fd.filter_out_non_face_corps(recreate_data=reload_images_from_db) # keeps only face-present images in data
    num_classes = 21 # num of unique classes

    if recreate_data:
        print('Creating X,y datasets based on day splits')
        X_train ,y_train, X_val ,y_val, X_test, y_test = fd.create_X_y_faces() # create an X,y dataset from filtered images
        print(f"Creates a label encoder and removes entered classes {num_classes} from dataset")
        X_train,y_train,X_val,y_val,X_test,y_test, le = labelencode(data_path,  X_train ,y_train, X_val ,y_val, X_test, y_test, [])
        # le = preprocessing.LabelEncoder()
        # le = le.fit([int(i) for cur_y_set in [y_train, y_val, y_test] for i in cur_y_set])
        # pickle.dump(le, open(os.path.join(data_path, 'le.pkl'), 'wb'))
        fc = FaceClassifer(num_classes, le, device='cuda:1')
        print("Creates data loaders")
        dl_train,dl_val,dl_test = fc.create_data_loaders(X_train ,y_train, X_val ,y_val, X_test, y_test, data_path)
    else:
        print('Loading data given path ')
        le, dl_train, dl_val, dl_test = load_data(data_path)
        fc = FaceClassifer(num_classes, le, device='cuda:1')
        if load_checkpoint:
            print('checkpoint path received, loading..')
            fc.model_ft.load_state_dict(torch.load(os.path.join(checkpoint_path,load_checkpoint)))

    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    print('Begin Training')
    try:
        fc.run_training(checkpoint_path=checkpoint_path, dl_train=dl_train, dl_val=dl_val,model_name=run_name,num_epochs=3) # train the model
        print('Training Done')
    except KeyboardInterrupt:
        print('Training was interrupted by user')
        test_accuracy_of_dataset(fc, dl_train, 'Train')
        test_accuracy_of_dataset(fc, dl_val, 'Val')
        test_accuracy_of_dataset(fc, dl_test, 'Test')

    test_accuracy_of_dataset(fc, dl_train, 'Train')
    test_accuracy_of_dataset(fc, dl_val, 'Val')
    test_accuracy_of_dataset(fc, dl_test, 'Test')

if __name__ == '__main__':
    data_path = '/mnt/raid1/home/bar_cohen/FaceData'
    # fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97])
    # fd.filter_out_non_face_corps(recreate_data=True) # keeps only face-present images in data
    # fd.create_X_y_faces()
    # checkpoint = ""
    checkpoint_path = os.path.join("/mnt/raid1/home/bar_cohen/FaceData/checkpoints/")
    main_train(data_path='/mnt/raid1/home/bar_cohen/FaceData/', run_name='check_le',reload_images_from_db=True, recreate_data=True,
               checkpoint_path=checkpoint_path, load_checkpoint='')

    # eval_faceClassifier(os.path.join(checkpoint_path, 'check_le, 3.pth'))
