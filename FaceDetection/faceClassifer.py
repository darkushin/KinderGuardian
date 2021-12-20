import copy
import pickle
import time
from collections import defaultdict

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms, utils, datasets, models
from FaceDetection.facenet_pytorch import MTCNN, InceptionResnetV1
from torch import nn, optim
from matplotlib import pyplot as plt
import torch
import numpy as np

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

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        print(self.model_ft)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=0.0001)
        # self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=7, gamma=0.1)


    def _create_Incepction_Resnet_for_finetunning(self):
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=False)
        layer_list = list(model_ft.children())[-5:] # final layers
        print(layer_list)
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

    def create_data_loaders(self, X, y) -> (DataLoader, DataLoader, DataLoader):
        # apply_transform  = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        plt.show()
        print('showed1')
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2 , random_state = 1)
        X_val, X_test, y_val, y_test = train_test_split(X_val,y_val, test_size = 0.5 , random_state = 1)
        plt.show()
        print('showed2')
        # print(type(X_train[0]), X_train[0].shape)
        X_train = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x) for x in X_train]
        X_val = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)for x in X_val]
        X_test = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x) for x in X_train]
        dl_train = DataLoader(FacesDataset(X_train, y_train), batch_size=100, shuffle=True)
        dl_val = DataLoader(FacesDataset(X_val,y_val), batch_size=100, shuffle=True)
        dl_test = DataLoader(FacesDataset(X_test,y_test), batch_size=100, shuffle=True)
        return dl_train, dl_val, dl_test

    def train_model(self,dl_train:DataLoader, dl_val:DataLoader, num_epochs=25):
        dataloaders = {'train': dl_train, 'val':dl_val}
        dataset_size = {'train': len(dl_train) , 'val': len(dl_val)}
        since = time.time()
        FT_losses = []

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
                            # self.exp_lr_scheduler.step()

                    FT_losses.append(loss.item())
                    # statistics
                    # print(preds, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    # if i % 1 == 0:
                    #     print('[%d, %5d] loss: %.3f' %
                    #           (epoch + 1, i + 1, np.mean(FT_losses)))

                    # print(running_corrects)
                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the self.model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model_ft.state_dict())
                torch.save(best_model_wts, 'best_model.pkl')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        self.model_ft.load_state_dict(best_model_wts)
        return self.model_ft, FT_losses

    def imshow(self, inp):
        """Imshow for Tensor."""
        plt.figure()
        f , ax = plt.subplots(len(inp),1)
        for i, img in enumerate(inp):
            img = img.permute(1,2,0).int().numpy()
            ax[i].imshow(img)
        plt.show()


if __name__ == '__main__':
    # fc = FaceDetector()
    # fc.filter_out_non_face_corps()
    # X,y = fc.create_X_y_faces()
    # pickle.dump(X, open("X.pkl",'wb'))
    # pickle.dump(y, open("y.pkl",'wb'))
    # print('done')
    X = pickle.load(open('X.pkl','rb'))
    y = pickle.load(open('y.pkl','rb'))
    y = [int(i) for i in y] # TODO fix y to hold ints and not strs
    le = preprocessing.LabelEncoder()
    y_transformed = le.fit_transform(y)
    num_classes = len(np.unique(y_transformed))
    print(num_classes)
    print(y)
    # print(X,y)
    fc = FaceClassifer(num_classes)
    fc.imshow(X[0:5])
    plt.show()

    x,y,z = fc.create_data_loaders(X,y_transformed)
    fc.train_model(x, y,num_epochs=3000)
    # fc.imshow(t)