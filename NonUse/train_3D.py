from NonUse import textneck_dataset_3D, adaptive_wing_loss
import textneck_test_dataset
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import os
import turtlenet_arch_3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=5
torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
#Train Dataloader & Test Dataloader
dataset = textneck_dataset_3D.TextneckDataset3D(root_dir='//', transform=torchvision_transform)
dataloader = textneck_dataset_3D.DataLoader(dataset, batch_size=batch_size, shuffle=False)

test_dataset = textneck_test_dataset.TextneckTestDataset(root_dir='//', transform=torchvision_transform)
test_dataloader = textneck_test_dataset.DataLoader(test_dataset, batch_size=1, shuffle=True)

#turtlenet = turtlenet_arch_2ch.TurtleNet(pretrained=False).to(device)
turtlenet = turtlenet_arch_3D.TurtleNet().to(device)

#Define Loss
criterition = adaptive_wing_loss.AdaptiveWingLoss()

#Define Optimizer
optimizer = optim.Adam(turtlenet.parameters(), lr=0.0005)

for epoch in range(1000):

    running_loss = 0.0

    for input, target in dataloader:

        optimizer.zero_grad()
        feature_output = turtlenet(input.to(device))
        loss = criterition(feature_output,target)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()

        #netron
        # params = turtlenet.state_dict()
        # dummy_data = torch.empty(1,3,256,256,dtype=torch.float32).to(device)
        # torch.onnx.export(turtlenet,dummy_data,"turtlenet_basic_block.onnx")

    # if epoch%2==0 and epoch!=0:
    #     with torch.no_grad():
    #         for input_test in test_dataloader:
    #
    #             feature_output_test=turtlenet(input_test.to(device))
    #
    #             fig = plt.figure()
    #             rows = 1
    #             cols = 2
    #
    #             ax1 = fig.add_subplot(rows, cols, 1)
    #             ax1.imshow(input_test[0].permute(1,2,0))
    #             ax1.set_title('Input image')
    #             ax1.axis("off")
    #
    #             ax2 = fig.add_subplot(rows, cols, 2)
    #             #+feature_output_test[0][1]
    #             ax2.imshow((feature_output_test[0][0]+feature_output_test[0][1]).cpu())
    #             ax2.set_title('Heatmap')
    #             ax2.axis("off")
    #
    #             plt.show()
    #             break;

    if (epoch+1)%100==0 and epoch!=0:
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': turtlenet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 'turtlenet_255_AdaptiveWingLoss_epoch_' + str(epoch+1) + '.pth')

            print("save complete")
        except:
            print("save error!")

    print("epoch:", epoch + 1, running_loss)

