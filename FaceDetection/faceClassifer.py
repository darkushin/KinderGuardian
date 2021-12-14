import pickle
from collections import defaultdict

from PIL import Image
from sklearn.model_selection import train_test_split

from DataProcessing.utils import read_labled_croped_images
from FaceDetection.facenet_pytorch import MTCNN, InceptionResnetV1
from torch import nn
from mtcnn.mtcnn import MTCNN as mtcnn_origin
from matplotlib import pyplot as plt
import torch


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = torch.F.normalize(x, p=2, dim=1)
        return x

class FaceClassifer():

    def __init__(self, num_classes):
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def _create_Incepction_Resnet_for_finetunning(self):

        model_ft = InceptionResnetV1(pretrained='vggface2', classify=False)
        layer_list = list(model_ft.children())[-5:] # final layers

        model_ft = torch.nn.Sequential(*list(model_ft.children())[:-5]) # skipping 5 last layers
        for param in model_ft.parameters():
            param.requires_grad = False # we dont want to train the original layers

        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        model_ft.last_linear = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=1792, out_features=512, bias=False),
            normalize()
        )

        model_ft.logits = nn.Linear(layer_list[3].in_features, len(self.num_classes))
        model_ft.softmax = nn.Softmax(dim=1)
        model_ft = model_ft.to(self.device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)
        # Decay LR by a factor of *gamma* every *step_size* epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

