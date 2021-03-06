import torch
import turtlenet_arch_2ch_A
import textneck_test_dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

device = torch.device("cuda")
turtlenet = turtlenet_arch_2ch_A.TurtleNet().to(device)
checkpoint = torch.load('C:/Users/user/TurtleNet/turtlenet_AdaptiveWingLoss_DC_epoch_175.pth')
turtlenet.load_state_dict(checkpoint['model_state_dict'])
turtlenet.eval()

print("TurtleNet load complete.")
###------WebCam------###
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
prev_time = 0
FPS = 10

while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    current_time = time.time() - prev_time
    if (ret) and (current_time > 1./ FPS) :
        prev_time = time.time()

        #cv2.imshow('frame_color', frame)# 컬러 화면 출력

        feature_output=turtlenet(torch.reshape(torchvision_transform(torch.tensor(frame)),(1,3,256,256)).to(device))

        #extract point

        #cv2.imshow('heat_map',((feature_output[0][0] + feature_output[0][1]).cpu()).detach().numpy())
        print(torch.where(feature_output[0][0] == torch.max(feature_output[0][0])))
        print(torch.where(feature_output[0][1] == torch.max(feature_output[0][1])))
        cv2.line(frame,(int(torch.where(feature_output[0][0] == torch.max(feature_output[0][0]))[0]),
                        int(torch.where(feature_output[0][0] == torch.max(feature_output[0][0]))[1])),
                 (int(torch.where(feature_output[0][1] == torch.max(feature_output[0][1]))[0]),
                  int(torch.where(feature_output[0][1] == torch.max(feature_output[0][1]))[1])),(0,0,255),5 )
        #cv2.imshow((feature_output[0][0] + feature_output[0][1]).cpu())
        cv2.imshow('frame_color', frame)
        # cv2.scatter(torch.where(feature_output[0][0] == torch.max(feature_output[0][0]))[1].cpu(),
        #             torch.where(feature_output[0][0] == torch.max(feature_output[0][0]))[0].cpu())
        # cv2.scatter(torch.where(feature_output[0][1] == torch.max(feature_output[0][1]))[1].cpu(),
        #             torch.where(feature_output[0][1] == torch.max(feature_output[0][1]))[0].cpu())
        # print(torch.max(feature_output[0][0]), torch.max(feature_output[0][1]))
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

###------Test_DataLoader------###
# test_dataset = textneck_test_dataset.TextneckTestDataset(root_dir='C:/Users/user/TurtleNet/', transform=torchvision_transform)
# test_dataloader = textneck_test_dataset.DataLoader(test_dataset, batch_size=1, shuffle=True)
#
# for i in range(50):
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
