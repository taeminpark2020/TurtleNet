from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import cv2
import turtlenet_arch_2ch
import torch.functional as F
from math import exp
# transforms
torchvision_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

device = torch.device("cuda")
turtlenet = turtlenet_arch_2ch.TurtleNet().to(device)
checkpoint = torch.load('C:/Users/user/TurtleNet/turtlenet_255_AdaptiveWingLoss_epoch_300.pth')
turtlenet.load_state_dict(checkpoint['model_state_dict'])
turtlenet.eval()

class TextneckDataset3D(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir =root_dir
        with open(self.root_dir+'textneck_3D_index.txt') as f:
            index = f.read()
            f.close()

        tmp = index.split("\n")
        self.index = tmp

        self.transform = transform

    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):

        image_data = np.stack((plt.imread(self.root_dir + 'TextNeckFrames/'+self.index[idx]),
                               plt.imread(self.root_dir + 'TextNeckFrames/'+self.index[idx]),
                               plt.imread(self.root_dir + 'TextNeckFrames/'+self.index[idx]),
                               plt.imread(self.root_dir + 'TextNeckFrames/'+self.index[idx]),
                               plt.imread(self.root_dir + 'TextNeckFrames/'+self.index[idx])))

        try:
            self.input = image_data

            if self.input.shape[3] == 1:
                self.input = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)

        except:
            self.input = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        if self.transform:
            self.target = turtlenet(self.transform(self.input[4]).view(-1,3,256,256).cuda())
            self.target = self.target[0]

            self.input = torch.stack([self.transform(self.input[0]),
                                      self.transform(self.input[1]),
                                      self.transform(self.input[2]),
                                      self.transform(self.input[3]),
                                      self.transform(self.input[4])],dim=1)

        return self.input, self.target

# dataset = TextneckDataset3D(root_dir='C:/Users/user/TurtleNet/', transform=torchvision_transform)
# dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
#
# for input, target in dataloader:
#
#     print(input.shape, target.shape)
#     fig = plt.figure()
#     rows = 1
#     cols = 3
#
#     ax1 = fig.add_subplot(rows, cols, 1)
#     ax1.imshow(input[0].permute(1,2,0))
#     ax1.set_title('Input image')
#     ax1.axis("off")
#
#     ax2 = fig.add_subplot(rows, cols, 2)
#     ax2.imshow(output_ear[0])
#     ax2.set_title('Output Ear Heatmap')
#     ax2.axis("off")
#
#     ax3 = fig.add_subplot(rows, cols, 3)
#     ax3.imshow(output_c7[0])
#     ax3.set_title('Output C7 Heatmap')
#     ax3.axis("off")
#
#     plt.show()
