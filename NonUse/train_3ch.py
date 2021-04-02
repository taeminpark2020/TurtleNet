import textneck_dataset
import textneck_test_dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import os
from NonUse import turtlenet_arch_3ch
import matplotlib.pyplot as plt
import torch.onnx

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=10
torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

dataset = textneck_dataset.TextneckDataset(root_dir='//', transform=torchvision_transform)
dataloader = textneck_dataset.DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = textneck_test_dataset.TextneckTestDataset(root_dir='//', transform=torchvision_transform)
test_dataloader = textneck_test_dataset.DataLoader(test_dataset, batch_size=1, shuffle=False)

turtlenet = turtlenet_arch_3ch.TurtleNet(pretrained=False).to(device)
# recon_ear = recon.ReconEar().to(device)
# recon_c7 = recon.ReconC7().to(device)

#Define Loss
criterition = nn.MSELoss(reduction="sum")

#Define Parameters
#parameters = list(resnet.parameters())+list(recon_ear.parameters())+list(recon_c7.parameters())

#Define Optimizer
#criterition_resnet = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(turtlenet.parameters(), lr=0.0005)

for epoch in range(1000):

    running_loss = 0.0

    for input, ear_target, c7_target in dataloader:

        optimizer.zero_grad()

        feature_output = turtlenet(input.to(device))
        # ear_pred = recon_ear(feature_output)
        # c7_pred = recon_c7(feature_output)

        loss = criterition(torch.stack((ear_target, c7_target, ear_target+c7_target),dim=1).to(device),feature_output.to(device))
        #loss = criterition((ear_target+c7_target).to(device), feature_output.view(-1,256,256).to(device))
        #loss = criterition(ear_pred, ear_target.to(device))+criterition(c7_pred, c7_target.to(device))
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
        #netron
        # params = turtlenet.state_dict()
        # dummy_data = torch.empty(1,3,256,256,dtype=torch.float32).to(device)
        #
        # torch.onnx.export(turtlenet,dummy_data,"turtlenet_basic_block.onnx")

    if epoch%25==0 and epoch!=0:
        with torch.no_grad():
            for input_test in test_dataloader:

                feature_output_test=turtlenet(input_test.to(device))

                fig = plt.figure()
                rows = 1
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(input_test[0].permute(1,2,0))
                ax1.set_title('Input image')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 2)
                #+feature_output_test[0][1]
                ax2.imshow((feature_output_test[0][2]).cpu())
                ax2.set_title('Heatmap')
                ax2.axis("off")

                plt.show()
                break;
    print("epoch:", epoch + 1, running_loss)

