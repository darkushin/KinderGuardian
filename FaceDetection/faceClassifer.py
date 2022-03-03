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
from FaceDetection.facenet_pytorch import InceptionResnetV1
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
    def __init__(self, num_classes, label_encoder):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        print(self.model_ft)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(),lr=0.0001, amsgrad=False)
        self.writer = SummaryWriter('runs/images')
        self.le = label_encoder
        # self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=7, gamma=0.1)


    def _create_Incepction_Resnet_for_finetunning(self):
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=True)
        layer_list = list(model_ft.children())[-5:] # final layers
        model_ft = torch.nn.Sequential(*list(model_ft.children())[:-5]) # skipping 5 last layers
        for param in model_ft.parameters():
            param.requires_grad = False # we dont want to train the original layers

        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        model_ft.last_linear = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=1792, out_features=512, bias=False),
            normalize())
        model_ft.logits = nn.Linear(layer_list[4].in_features, self.num_classes)
        model_ft.softmax = nn.Softmax(dim=1)
        model_ft = model_ft.to(self.device)
        return model_ft

    def create_data_loaders(self, X, y, data_path) -> (DataLoader, DataLoader, DataLoader):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1 , shuffle=True, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2 , shuffle=True, random_state=1)
        dl_train = DataLoader(FacesDataset(X_train, y_train), batch_size=100, shuffle=True)
        dl_val = DataLoader(FacesDataset(X_val,y_val), batch_size=10, shuffle=True)
        dl_test = DataLoader(FacesDataset(X_test,y_test), batch_size=10, shuffle=True)
        # must save datasets dls for reproducibility
        pickle.dump(dl_train, open(os.path.join(data_path, 'dl_train_1.pkl'), 'wb'))
        pickle.dump(dl_val, open(os.path.join(data_path, 'df_val_1.pkl'), 'wb'))
        pickle.dump(dl_test, open(os.path.join(data_path, 'df_test_1.pkl'), 'wb'))
        return dl_train, dl_val, dl_test

    def train_model(self,dl_train:DataLoader, dl_val:DataLoader, num_epochs=25):
        """ Main training loop for model"""
        dataloaders = {'train': dl_train, 'val':dl_val}
        dataset_size = {'train': len(dl_train) , 'val': len(dl_val)}
        since = time.time()
        train_losses = []
        val_losses = []
        acc_by_epoch_train = []
        acc_by_epcoh_val = []
        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model_ft.train()  # Set self.model to training mode
                else:
                    self.model_ft.eval()  # Set self.model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for i, batch in enumerate(dataloaders[phase]):
                    inputs , labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer_ft.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()
                            train_losses.append(loss.item())
                        else:
                            val_losses.append(loss.item())
                            # self.exp_lr_scheduler.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds.eq(labels.data)).item()
                    # if i % 1 == 0:
                    #     print('[%d, %5d] loss: %.3f' %
                    #           (epoch + 1, i + 1, np.mean(FT_losses)))
                    # todo Tensor board upgrade WIP
                    img_grid = make_grid(inputs)
                    self.writer.add_image('image_input', img_grid)

                    if i == len(dataloaders[phase]) - 1 : # last batch
                        named_preds = [ID_TO_NAME[self.le.inverse_transform([i.item()])[0]]for i in preds[0:5]]
                        named_labels = [ID_TO_NAME[self.le.inverse_transform([i.item()])[0]] for i in labels[0:5]]
                        print(f'phase: {phase} prediction: {named_preds}, true_labels: {named_labels}')
                        # self.imshow(inputs[0:5], named_preds)
                        # self.writer.add_graph(self.model_ft, inputs)

                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects / (dataset_size[phase] * dataloaders[phase].batch_size)
                if phase == 'train':
                    acc_by_epoch_train.append(epoch_acc)
                else:
                    acc_by_epcoh_val.append(epoch_acc)

                print('phase:{} ,  Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))

            # deep copy the self.model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model_ft.state_dict())
                torch.save(best_model_wts, os.path.join("/mnt/raid1/home/bar_cohen/FaceData/", 'best_model4.pkl'))
            metrics = {'batch_train_loss': train_losses, 'batch_val_loss': val_losses,
                       'train_acc_by_epoch': acc_by_epoch_train, 'val_acc_by_epoch': acc_by_epcoh_val}
            if epoch > 1:
                # fc.plot_results(metrics)
                pass
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        self.model_ft.load_state_dict(best_model_wts)
        metrics = {'batch_train_loss': train_losses, 'batch_val_loss': val_losses,
                   'train_acc_by_epoch': acc_by_epoch_train, 'val_acc_by_epoch': acc_by_epcoh_val}
        self.writer.close()
        return self.model_ft, metrics

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
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs)
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


def test_accuracy_of_dataset(fc, dl, name):
    """ Simple accuracy tester for face Classfication """
    print('Accuracy of dataset --- ', name)
    acc = []
    for batch in dl:
        images, labels = batch
        inputs = images.to(fc.device)
        labels = labels.to(fc.device)
        preds, _ = fc.predict(inputs)
        acc.append(fc.get_accuracy(preds, labels))
    print(np.mean(acc))

def labelencode(label_encoder_output,X,y ,classes_to_drop:list):
    """
    creates a label encoder based on inserted classes to drop
    Args:
        label_encoder_output: output path to save label encoder
        X: the image tensors
        y: image labels
        classes_to_drop: label classes not to be trained upon

    Returns: X_transformed, y_transformed, label-encoder

    """
    y = [int(i) for i in y]
    indexs_to_drop = [index for index, clss in enumerate(y) if clss in classes_to_drop]
    X = [x for i, x in enumerate(X) if i not in indexs_to_drop]
    y = [clss for i, clss in enumerate(y) if i not in indexs_to_drop]
    assert len(X) == len(y)
    le = preprocessing.LabelEncoder()
    y_transformed = le.fit_transform(y)
    pickle.dump(le, open(os.path.join(label_encoder_output, 'le.pkl'), 'wb'))
    return X, y_transformed, le

def load_data(data_path):
    le = pickle.load(open(os.path.join(data_path, 'le.pkl'), 'rb'))
    dl_train = pickle.load(open(os.path.join(data_path, 'dl_train_1.pkl'), 'rb'))
    dl_val = pickle.load(open(os.path.join(data_path, 'df_val_1.pkl'), 'rb'))
    dl_test = pickle.load(open(os.path.join(data_path, 'df_test_1.pkl'), 'rb'))
    return le, dl_train, dl_val, dl_test

def create_face_data_from_db():
    face_crops = get_entries(filters={Crop.is_face == True,
                                      Crop.reviewed_one == True ,
                                      Crop.is_vague == False ,
                                      Crop.invalid == False}).all()
    crops_path = "/mnt/raid1/home/bar_cohen/"
    data_path = '/mnt/raid1/home/bar_cohen/FaceData/images/'
    for crop in face_crops:
        if crop.vid_name[4:8] not in ['0808','3007']:
            shutil.copy(os.path.join(crops_path, crop.vid_name, crop.im_name), os.path.join(data_path, crop.im_name))
    print('done')
    # y = [NAME_TO_ID[crop.label] for crop in face_crops]
    # return X, y

def main_train():
    """
    This is an example of a training pipeline. Insert a raw images path or existing
    faces data path to build a dataset and train a faceClassifer on it. Data will be saved
    as well metrics for the training.
    Returns:

    """
    # create_face_data_from_db()
    # return
    data_path = '/mnt/raid1/home/bar_cohen/FaceData/'
    # fd = FaceDetector(raw_images_path='/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned',
    #                   faces_data_path='C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl') # init faceDetector
    fd = FaceDetector()
    #
    fd.filter_out_non_face_corps() # keeps only face-present images in data
    X,y = fd.create_X_y_faces() # create an X,y dataset from filtered images
    # X,y_transformed,le = labelencode(data_path, X,y,[17,19]) # creates a label encoder and removes entered classes from dataset
    # num_classes = len(np.unique(y_transformed)) # num of unique classes
    le, dl_train, dl_val, dl_test = load_data(data_path)
    fc = FaceClassifer(19, le)  # init faceClassifer
    # loads an already train model to keep training it - uncomment if you want to train from scratch
    fc.model_ft.load_state_dict(torch.load(os.path.join(data_path,'best_model3.pkl')))
    # fc.model_ft.train() # this will train the model after loading weights
    # creates and saves datasets based on the lines above,
    # if you want to train based on exsisting data split, load it first
    # dl_train,dl_val,dl_test = fc.create_data_loaders(X,y_transformed, data_path)

    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    model, metrics = fc.train_model(dl_train, dl_val,num_epochs=1000) # train the model
    # run acc test on data splits
    test_accuracy_of_dataset(fc, dl_train, 'Train')
    test_accuracy_of_dataset(fc, dl_val, 'Val')
    test_accuracy_of_dataset(fc, dl_test, 'Test')
    # save metrics for later lookup
    pickle.dump(metrics , open(os.path.join(data_path, 'best_new_metric_1.pkl'),'wb'))
    # plot metrics results
    fc.plot_results(metrics)


if __name__ == '__main__':
    main_train()