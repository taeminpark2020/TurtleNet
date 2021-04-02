import textneck_dataset
import textneck_test_dataset
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import os
import turtlenet_arch_multiatn
import matplotlib.pyplot as plt
import visdom
import DiceLoss
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=10
epochs = 3000
torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
#Train Dataloader & Test Dataloader
dataset = textneck_dataset.TextneckDataset(root_dir=os.path.abspath(os.path.join(''))+'/', transform=torchvision_transform)
dataloader = textneck_dataset.DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = textneck_test_dataset.TextneckTestDataset(root_dir=os.path.abspath(os.path.join(''))+'/', transform=torchvision_transform)
test_dataloader = textneck_test_dataset.DataLoader(test_dataset, batch_size=1, shuffle=True)

#turtlenet = turtlenet_arch_2ch.TurtleNet(pretrained=False).to(device)
#turtlenet = turtlenet_arch_2ch_A.TurtleNet().to(device)
turtlenet = turtlenet_arch_multiatn.TurtleNet(pretrained=False).to(device)

#Define Loss
criterition = DiceLoss.BinaryDiceLoss()
#Define Optimizer
optimizer = optim.Adam(turtlenet.parameters(), lr=0.0005)

#Visdom Visualization
vis = visdom.Visdom()

# checkpoint = torch.load('C:/Users/user/TurtleNet/turtlenet_300.pth')
# turtlenet.load_state_dict(checkpoint['model_state_dict'])
#turtlenet.load_state_dict(checkpoint['optimizer_state_dict'])
turtlenet.train()


print('\033[32m'+"======================================================================")
print('\033[32m'+"Train_image_dataset_Num : " + str(len(os.walk(os.path.abspath(os.path.join(''))+'/TextNeckImage').__next__()[2])))
print('\033[32m'+"Test_image_dataset_Num : " + str(len(os.walk(os.path.abspath(os.path.join(''))+'/TextNeckImage_Test').__next__()[2])))
print('\033[32m'+"Batch_size : "+str(batch_size))
print('\033[32m'+"Eopch : "+str(epochs))
print('\033[32m'+"======================================================================")
print('\033[97m'+"Training_Start")

for epoch in range(epochs):

    running_loss = 0.0

    for input, ear_target, c7_target in dataloader:

        optimizer.zero_grad()
        feature_output = turtlenet(input.to(device))
        loss = criterition(feature_output.to(device),torch.stack((ear_target, c7_target),dim=1).to(device))

        loss.backward()
        optimizer.step()
        running_loss +=loss.item()

        #netron
        # params = turtlenet.state_dict()
        # dummy_data = torch.empty(1,3,256,256,dtype=torch.float32).to(device)
        # torch.onnx.export(turtlenet,dummy_data,"turtlenet_basic_block.onnx")
    if epoch == 0:
        win = vis.line(X=[epoch+1], Y=[running_loss])
    else:
        vis.line(X=[epoch+1],Y=[running_loss], win=win, update='append')
    print("epoch :", epoch + 1, running_loss)

    #Validation
    if (epoch+1)%25==0 and epoch!=0:

        with torch.no_grad():
            running_loss_test = 0.0
            for input_test, ear_test, c7_test in test_dataloader:

                feature_output_test=turtlenet(input_test.to(device))

                #just one time visualization
                if running_loss_test==0.0:
                    #fig.imshow(input_test[0].permute(1, 2, 0))
                    # fig.scatter(int(torch.where(feature_output_test[0][1] == torch.max(feature_output_test[0][1]))[1]),
                    #             int(torch.where(feature_output_test[0][1] == torch.max(feature_output_test[0][1]))[0]))

                    ear_point=torch.where(feature_output_test[0][0] == torch.max(feature_output_test[0][0]))
                    c7_point=torch.where(feature_output_test[0][1] == torch.max(feature_output_test[0][1]))

                    # input_test[0][0][ear_point[1][0]-20:ear_point[1][0]+20][ear_point[0][0]-20:ear_point[0][0]+20]=0
                    # input_test[0][1][ear_point[1][0] - 20:ear_point[1][0] + 20][ear_point[0][0] - 20:ear_point[0][0] + 20] = 0
                    # input_test[0][2][ear_point[1][0] - 20:ear_point[1][0] + 20][ear_point[0][0] - 20:ear_point[0][0] + 20] = 1
                    # input_test[0][0][c7_point[1][0]-20:c7_point[1][0]+20][c7_point[0][0]-20:c7_point[0][0]+20]=0
                    # input_test[0][1][c7_point[1][0] - 20:c7_point[1][0] + 20][c7_point[0][0] - 20:c7_point[0][0] + 20] = 0
                    # input_test[0][2][c7_point[1][0] - 20:c7_point[1][0] + 20][c7_point[0][0] - 20:c7_point[0][0] + 20] = 1

                    vis.image(input_test[0],opts=dict(title="Input",caption="Input"))

                    # vis.heatmap(X=np.flipud(feature_output_test[0][0].cpu()),opts=dict(title = "Ear",
                    #                      caption = "Ear_Heatmap"))
                    # vis.heatmap(X=np.flipud(feature_output_test[0][1].cpu()),opts=dict(title = "C7",
                    #                      caption = "C7_Heatmap"))
                    vis.heatmap(X=np.flipud((feature_output_test[0][0]+feature_output_test[0][1]).cpu()),
                              opts=dict(title = "Ear+C7",caption = "Ear + C7_Heatmap"))
                break;
                # loss_test = criterition(feature_output_test.to(device), torch.stack((ear_test, c7_test), dim=1).to(device))
                # running_loss_test += loss_test.item()

            # if epoch == 4:
            #     win_accuracy = vis.line(X=[epoch + 1], Y=[100-(running_loss_test/680)*100])
            # else:
            #     vis.line(X=[epoch + 1], Y=[running_loss], win=win_accuracy, update='append')

    if (epoch+1)%50==0 and epoch!=0:
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': turtlenet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 'turtlenet_dice+' + str(epoch+1) + '.pth')

            print("save complete")
        except:
            print("save error!")

