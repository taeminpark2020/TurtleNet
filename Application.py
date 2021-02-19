import torch
import turtlenet_arch_2ch
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
turtlenet = turtlenet_arch_2ch.TurtleNet().to(device)
checkpoint = torch.load('C:/Users/user/TurtleNet/turtlenet_epoch_100.pth')
turtlenet.load_state_dict(checkpoint['model_state_dict'])
turtlenet.eval()

print("TurtleNet load complete.")

# cap = cv2.VideoCapture(0)
# prev_time = 0
# FPS = 3
#
# while(True):
#     ret, frame = cap.read()    # Read 결과와 frame
#
#     current_time = time.time() - prev_time
#     if (ret) and (current_time > 1./ FPS) :
#         prev_time = time.time()
#
#         cv2.imshow('frame_color', frame)# 컬러 화면 출력
#
#         feature_output=turtlenet(torch.reshape(torchvision_transform(torch.tensor(frame)),(1,3,256,256)).to(device))
#
#         cv2.imshow('heat_map',((feature_output[0][0] + feature_output[0][1]).cpu()).detach().numpy())
#
#         if cv2.waitKey(1) == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()






#
test_dataset = textneck_test_dataset.TextneckTestDataset(root_dir='C:/Users/user/TurtleNet/', transform=torchvision_transform)
test_dataloader = textneck_test_dataset.DataLoader(test_dataset, batch_size=1, shuffle=True)

for i in range(50):
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
            ax2.imshow((feature_output_test[0][0]+feature_output_test[0][1]).cpu())
            ax2.set_title('Heatmap')
            ax2.axis("off")

            plt.show()
            break;