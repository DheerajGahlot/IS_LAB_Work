# Design and development of an UI for a Yoga Mat

We are using publicly avialable data set and we did classification on that 

As mentioned above, this dataset is publicly available, so we will download it here and rename the folder as "dataset":

Use following Command :-
!wget https://physionet.org/static/published-projects/pmd/a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0.zip
!unzip -q a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0.zip
!mv a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0 dataset 

Libraries we are using  
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

after installing all libraries simply run the mention code we can see the output 
