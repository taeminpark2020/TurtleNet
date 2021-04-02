from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import random
import cv2

import custom_transforms

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

        img_size = 39
        self.img_size = img_size
        scaledGaussian = lambda x : exp(-(1/2)*(x**2))
        isotropicGrayscaleImage = np.zeros((img_size, img_size), dtype='float')

        sigma = 5

        for i in range(img_size):
            for j in range(img_size):
                distanceFromCenter = np.linalg.norm(np.array([i-img_size/2, j-img_size/2]))
                distanceFromCenter =distanceFromCenter/sigma
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                isotropicGrayscaleImage[i, j] = np.clip(scaledGaussianProb, 0, 1)

        self.heatmap = (torch.tensor(isotropicGrayscaleImage))

        #print(self.heatmap, torch.max(self.heatmap))
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        self.random = random.randint(0,180)

        with open(self.root_dir+'TextNeckLabel/'+self.index[idx]+'.json') as f:
            json_data = json.load(f)
            f.close()
        image_data = plt.imread(self.root_dir+'TextNeckImage/'+self.index[idx])

        try:
            self.input = image_data

            if self.input.shape[2] == 1:
                self.input = cv2.cvtColor(np.float32(image_data), cv2.COLOR_GRAY2RGB)

        except:
            self.input = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        #self.input = image_data

        if self.transform:
            #self.input = self.transform(self.input)
            self.input = custom_transforms.ToTensor()(
                custom_transforms.RandomRotation(
                    self.random)(
                    custom_transforms.GaussianBlur(9, sigma=(0.1, 5.0))(
                    custom_transforms.Resize((256,256))(
                    custom_transforms.ToPILImage()(
                        self.input)))))

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
            ear_heatmap[(int(torch.round(ear_coordinate[1]*(256/self.output['size']['height'])))-1) -int(self.img_size/2):(int(torch.round(ear_coordinate[1]*(256/self.output['size']['height'])))-1)+int(self.img_size/2)+1,
                        (int(torch.round(ear_coordinate[0]*(256/self.output['size']['width'])))-1) -int(self.img_size/2):(int(torch.round(ear_coordinate[0]*(256/self.output['size']['width'])))-1)+int(self.img_size/2)+1]=self.heatmap
        except:
            ear_heatmap=torch.zeros(int(512),int(512))
            ear_heatmap[(int(torch.round(ear_coordinate[1] * (256 / self.output['size']['height']))) - 1)+128 -int(self.img_size/2):(int(
                torch.round(ear_coordinate[1] * (256 / self.output['size']['height']))) - 1) + 128 +int(self.img_size/2)+1,
            (int(torch.round(ear_coordinate[0] * (256 / self.output['size']['width']))) - 1) +128 -int(self.img_size/2):(int(
                torch.round(ear_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 +int(self.img_size/2)+1] = self.heatmap
            ear_heatmap=ear_heatmap[128:384,128:384]

        c7_coordinate = torch.tensor(self.output['objects'][0]['nodes']['226b28ac-ae41-4cdf-b841-a4e3f820e527']['loc'])
        #int(c7_coordinate[1])-1 : H
        #int(c7_coordinate[0])-1 : W

        try:
            c7_heatmap[(int(torch.round(c7_coordinate[1]*(256/self.output['size']['height'])))-1)-int(self.img_size/2):(int(torch.round(c7_coordinate[1]*(256/self.output['size']['height'])))-1)+int(self.img_size/2)+1,
                       (int(torch.round(c7_coordinate[0]*(256/self.output['size']['width'])))-1)-int(self.img_size/2):(int(torch.round(c7_coordinate[0]*(256/self.output['size']['width'])))-1)+int(self.img_size/2)+1]=self.heatmap
        except:
            c7_heatmap = torch.zeros(int(512), int(512))
            c7_heatmap[(int(torch.round(c7_coordinate[1] * (256 / self.output['size']['height']))) - 1)+128 -int(self.img_size/2):(int(
                torch.round(c7_coordinate[1] * (256 / self.output['size']['height']))) - 1) + 128 +int(self.img_size/2)+1,
            (int(torch.round(c7_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 -int(self.img_size/2):(int(
                torch.round(c7_coordinate[0] * (256 / self.output['size']['width']))) - 1) + 128 +int(self.img_size/2)+1] = self.heatmap
            c7_heatmap =c7_heatmap[128:384,128:384]


        stacks = custom_transforms.ToTensor()(
            custom_transforms.RandomRotation(self.random)(
                custom_transforms.Resize((256,256))(
                    custom_transforms.ToPILImage()(torch.stack([ear_heatmap,ear_heatmap,c7_heatmap],0)))))

        self.output_ear = stacks[0]
        self.output_c7 = stacks[2]

        #Resize 256x256
        return self.input, self.output_ear, self.output_c7

#transforms
torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# dataset = TextneckDataset(root_dir='C:/Users/user/TurtleNet/', transform=torchvision_transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
# for input, output_ear, output_c7 in dataloader:
#     print(output_ear[0][0][0])
#     fig = plt.figure()
#     rows = 1
#     cols = 3
#
#     ax1 = fig.add_subplot(rows, cols, 1)
#     ax1.imshow(input[0].permute(1,2,0))
#     ax1.scatter(int(torch.where(output_ear[0] == torch.max(output_ear[0]))[1][0]),
#                 int(torch.where(output_ear[0] == torch.max(output_ear[0]))[0][0]))
#     ax1.scatter(int(torch.where(output_c7[0] == torch.max(output_c7[0]))[1][0]),
#                 int(torch.where(output_c7[0] == torch.max(output_c7[0]))[0][0]))
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
