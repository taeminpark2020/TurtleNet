from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import cv2
import torch.functional as F
from math import exp

'''
input : color frame(256, 256)
output : Heatmap of ear & c7 coordinate(256, 256)
'''

class TextneckDataset (Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        with open(self.root_dir+'textneck_index.txt') as f:
            index = f.read()
            f.close()

        tmp = index.split("\n")
        self.index = tmp
        self.transform = transform

        img_size = 27
        #img_size = 25
        scaledGaussian = lambda x : exp(-(1/2)*(x**2))
        isotropicGrayscaleImage = np.zeros((img_size, img_size), dtype='float')

        for i in range(img_size):
            for j in range(img_size):
                distanceFromCenter = np.linalg.norm(np.array([i-img_size/2, j-img_size/2]))
                #distanceFromCenter = 2.5*distanceFromCenter/(img_size/2)
                distanceFromCenter = 5*distanceFromCenter/(img_size/2)
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                isotropicGrayscaleImage[i, j] = np.clip(scaledGaussianProb*255, 0, 255)

        self.heatmap = (torch.tensor(isotropicGrayscaleImage))/256

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        with open(self.root_dir+'TextNeckLabel/'+self.index[idx]+'.json') as f:
            json_data = json.load(f)
            f.close()
        image_data = plt.imread(self.root_dir+'TextNeckImage/'+self.index[idx])

        try:
            self.input = image_data

            if list(self.input.shape)[2] == 1:
                self.input = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)

        except:
            self.input = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        #self.input = image_data

        if self.transform:
            self.input = self.transform(self.input)

        self.output = json_data

        '''
        ear = 'a58833ac-5d69-4d54-9e1d-542d016ca326'
        c7 = '226b28ac-ae41-4cdf-b841-a4e3f820e527'
        !caution!
        Basic rule : H(height)xW(width), Height first
        Frame starting index is 1
        Tensor starting index is 0, you need Frame coordinate - 1        
        '''

        ear_heatmap=torch.zeros(int(256),int(256))
        c7_heatmap=torch.zeros(int(256),int(256))

        ear_coordinate = torch.tensor(self.output['objects'][0]['nodes']['a58833ac-5d69-4d54-9e1d-542d016ca326']['loc'])

        #int(ear_coordinate[1])-1 : H
        #int(ear_coordinate[0])-1 : W
        try:
            ear_heatmap[(int(torch.round(ear_coordinate[1]*(256/self.output['size']['height'])))-1) -13:(int(torch.round(ear_coordinate[1]*(256/self.output['size']['height'])))-1)+14,
                        (int(torch.round(ear_coordinate[0]*(256/self.output['size']['width'])))-1) -13:(int(torch.round(ear_coordinate[0]*(256/self.output['size']['width'])))-1)+14]=self.heatmap
        except:
            ear_heatmap=torch.zeros(int(512),int(512))
            ear_heatmap[(int(torch.round(ear_coordinate[1] * (256 / self.output['size']['height']))) - 1)+128 -13:(int(
                torch.round(ear_coordinate[1] * (256 / self.output['size']['height']))) - 1) + 128 +14,
            (int(torch.round(ear_coordinate[0] * (256 / self.output['size']['width']))) - 1) +128 -13:(int(
                torch.round(ear_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 +14] = self.heatmap
            ear_heatmap=ear_heatmap[128:384,128:384]

        c7_coordinate = torch.tensor(self.output['objects'][0]['nodes']['226b28ac-ae41-4cdf-b841-a4e3f820e527']['loc'])
        #int(c7_coordinate[1])-1 : H
        #int(c7_coordinate[0])-1 : W

        try:
            c7_heatmap[(int(torch.round(c7_coordinate[1]*(256/self.output['size']['height'])))-1)-12:(int(torch.round(c7_coordinate[1]*(256/self.output['size']['height'])))-1)+13,
                       (int(torch.round(c7_coordinate[0]*(256/self.output['size']['width'])))-1)-12:(int(torch.round(c7_coordinate[0]*(256/self.output['size']['width'])))-1)+13]=self.heatmap
        except:
            c7_heatmap = torch.zeros(int(512), int(512))
            c7_heatmap[(int(torch.round(c7_coordinate[1] * (256 / self.output['size']['height']))) - 1)+128 -13:(int(
                torch.round(c7_coordinate[1] * (256 / self.output['size']['height']))) - 1) + 128 +14,
            (int(torch.round(c7_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 -13:(int(
                torch.round(c7_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 +14] = self.heatmap
            c7_heatmap =c7_heatmap[128:384,128:384]

        self.output_ear = ear_heatmap
        self.output_c7 = c7_heatmap

        #Resize 256x256
        return self.input, self.output_ear, self.output_c7

#transforms
torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# dataset = TextneckDataset(root_dir='/home/ptm0228/PycharmProjects/longstone/', transform=torchvision_transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
# for input, output_ear, output_c7 in dataloader:
#     print(input ,output_ear.size(), output_c7.size())
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
