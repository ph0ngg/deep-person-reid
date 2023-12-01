import numpy as np
import cv2
import torch
from .yolo_pose_onnx import *
import torch.nn as nn
import torch.nn.functional as F
import sys, os


width, height = 256, 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_image(img):
    #img = cv2.imread(filename)

    #print(img.shape)
    pred = read_model(img)
    img = cv2.resize(img, (width, height))
    pred[:, 0], pred[:, 2] = pred[:, 0] / 640.0 * width, pred[:,2] / 640.0 * width
    pred[:, 1], pred[:, 3] = pred[:,1] / 640.0 * height, pred[:,3] / 640.0 * height
    for i in range(6, 57):
        if i % 3 == 2:
            pred[:, i-2] = pred[:, i-2] / 640.0 * width
            pred[:, i-1] = pred[:, i-1] / 640.0 * height
    new_preds = []
    for rows in pred:
        if rows[4] == max(pred[:, 4]):
            new_preds.append(rows)
    new_preds = np.array(new_preds).astype(np.int64)
    return new_preds

def crop_and_resize(img, coor, new_width, new_height):
    for i in range(3):
      if coor[i] < 0:
        coor[i] = 0
    crop_img = img[coor[2]:coor[3], coor[0]:coor[1]]
    rz_img = cv2.resize(crop_img, (new_width, new_height))
    return rz_img

# img = 'D:\Phong\TT\market1501\Market-1501-v15.09.15\gt_bbox\\0001_c1s1_002301_00.jpg'
# pred = read_image(img)
# img = cv2.imread(img)
# img = cv2.resize(img, (width, height))
def torch_from_numpy(x):
    x = x.transpose(0, 3, 1, 2) # batchsize, height, width, channel --> batchsize, channel, height, width
    x = torch.from_numpy(x)
    x.requires_grad = True
    return x

def black_img(width, height):
    x = np.zeros((height, width, 3))
    return x.tolist()

def pose(img):
    img = img.detach().cpu().numpy()
    crop_head_img, crop_larm_img, crop_rarm_img, crop_lleg_img, crop_rleg_img, crop_body_img = [], [], [], [], [], []
    for i in range(img.shape[0]):
        img1 = np.transpose(img[i], (1, 2, 0))
        pred = read_image(img1)
        img1 = cv2.resize(img1, (width, height))

        for i in range(len(pred)):
            crop_head = [max(pred[i, 6], 30)-30, pred[i, 6]+30, max(pred[i, 7], 30)-30, pred[i, 7]+30]
            crop_leftarm = [min(pred[i, 24], pred[i, 30], pred[i, 36])-10, max(pred[i, 24], pred[i, 30], pred[i, 36])+10, 
                                min(pred[i, 25], pred[i, 31], pred[i,37])-10, max(pred[i, 25], pred[i, 31], pred[i, 37])+10] 
            crop_rightarm = [min(pred[i, 21], pred[i, 27], pred[i, 33])-10, max(pred[i, 21], pred[i, 27], pred[i, 33])+10, 
                                min(pred[i, 22], pred[i, 28], pred[i, 34])-10, max(pred[i, 22], pred[i, 28], pred[i, 34])+10]
            crop_body = [min(pred[i, 21], pred[i, 24], pred[i, 39], pred[i, 42])-10, max(pred[i, 21], pred[i, 24], pred[i, 39])+10,
                                min(pred[i, 22], pred[i, 25], pred[i, 40])-10, max(pred[i, 22], pred[i,25], pred[i, 40])+10]
            crop_leftleg = [min(pred[i, 42], pred[i, 48], pred[i, 54])-10, max(pred[i, 42], pred[i, 48], pred[i, 54])+10,
                                min(pred[i, 43], pred[i, 49], pred[i, 55])-10, max(pred[i, 43], pred[i, 49], pred[i, 55])+10]
            crop_rightleg = [min(pred[i, 39], pred[i, 45], pred[i, 51])-10, max(pred[i, 39], pred[i, 45], pred[i, 51])+10,
                                min(pred[i, 40], pred[i, 46], pred[i, 52])-10, max(pred[i, 40], pred[i, 46], pred[i, 52])+10]
            # print('crop_head', crop_head)
            # print('crop_leftarm', crop_leftarm)
            # print('crop_rightarm', crop_rightarm)
            # print('crop_body', crop_body)
            # print('crop_leftleg', crop_leftleg)
            # print('crop_rightleg', crop_rightleg)

            crop_head_img.append(crop_and_resize(img1, crop_head, 128, 128))
            crop_larm_img.append(crop_and_resize(img1, crop_leftarm, 64, 256))
            crop_rarm_img.append(crop_and_resize(img1, crop_rightarm, 64, 256))
            crop_body_img.append(crop_and_resize(img1, crop_body, 128, 256))
            crop_lleg_img.append(crop_and_resize(img1, crop_leftleg, 64, 256))
            crop_rleg_img.append(crop_and_resize(img1, crop_rightleg, 64, 256))
    lenn = len(crop_head_img)
    for i in range(8 - lenn):
        crop_head_img.append(black_img(128, 128))
        crop_larm_img.append(black_img(64, 256))
        crop_rarm_img.append(black_img(64, 256))
        crop_body_img.append(black_img(128, 256))
        crop_lleg_img.append(black_img(64, 256))
        crop_rleg_img.append(black_img(64, 256))   
    

    crop_head_img = np.stack(np.array(crop_head_img), axis= 0)
    crop_larm_img = np.stack(np.array(crop_larm_img), axis= 0)
    crop_rarm_img = np.stack(np.array(crop_rarm_img), axis= 0)
    crop_body_img = np.stack(np.array(crop_body_img), axis= 0)
    crop_lleg_img = np.stack(np.array(crop_lleg_img), axis= 0)
    crop_rleg_img = np.stack(np.array(crop_rleg_img), axis= 0)

    return torch_from_numpy(crop_head_img).float().to(device), torch_from_numpy(crop_larm_img).float().to(device), torch_from_numpy(crop_rarm_img).float().to(device), torch_from_numpy(crop_body_img).float().to(device), torch_from_numpy(crop_lleg_img).float().to(device), torch_from_numpy(crop_rleg_img).float().to(device)
        

#     cv2.imwrite('head.jpg', crop_head_img)
#     cv2.imwrite('body.jpg', crop_body_img)
#     cv2.imwrite('leftarm.jpg', crop_larm_img)

# # class SpatialTransformerNetBody(nn.Module):
# #     def __init__(self, size):
# #         super(SpatialTransformerNetBody, self).__init__()

# #         # Define the body cropping layer

# #         self.downsampled_body = nn.Sequential(
# #             nn.AvgPool2d(kernel_size=4, stride=4),
# #         )
# #         #256x64
# #         self.st_body_conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0)
# #         self.st_body_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.st_body_relu1 = nn.ReLU()
# #         #126x30
# #         self.st_body_conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=0)
# #         self.st_body_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
# #         self.st_body_relu2 = nn.ReLU()
# #         #61x13
# #         self.st_body_ip1 = nn.Linear(20 * 61 * 13, 20)
# #         self.st_body_relu3 = nn.ReLU()

# #         self.st_body_theta = nn.Linear(20, 6)
# #         self.st_body_layer = SpatialTransformer()

# #     def forward(self, crop_body, x_size2):
# #         # Body cropping and downsample
# #         downsampled_body = self.downsampled_body(crop_body)

# #         # Continue with the rest of the layers
# #         st_body_conv1 = self.st_body_conv1(downsampled_body)
# #         st_body_pool1 = self.st_body_pool1(st_body_conv1)
# #         st_body_relu1 = self.st_body_relu1(st_body_pool1)

# #         st_body_conv2 = self.st_body_conv2(st_body_relu1)
# #         st_body_pool2 = self.st_body_pool2(st_body_conv2)
# #         st_body_relu2 = self.st_body_relu2(st_body_pool2)

# #         st_body_ip1 = self.st_body_ip1(st_body_relu2.view(-1, 20 * 61 * 13))
# #         st_body_relu3 = self.st_body_relu3(st_body_ip1)

# #         st_body_theta = self.st_body_theta(st_body_relu3)
# #         st_body_output = self.st_body_layer(crop_body, st_body_theta)

# #         return st_body_output

# class SpatialTransformer(nn.Module):
#     def forward(self, x, theta):
#         grid = F.affine_grid(theta, x.size())
#         x_transformed = F.grid_sample(x, grid, align_corners= True)
#         return x_transformed


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#         # Slice layer
#         self.slice = lambda x: (x[:, :1024, :, :], x[:, 1024:, :, :])

#         # Local Fully Connected layer
#         self.simpleconnection = nn.Linear(1024, 1)

#         # Tanh activation layer
#         self.tanhout = nn.Tanh()

#         # Reshape layer
#         self.reglobal = lambda x: x.view(x.size(0), -1)

#         # Concatenate layer
#         self.concat = lambda x, y: torch.cat((x, y), dim=1)

#         # Fully Connected layer (ip2)
#         self.ip2 = nn.Linear(2048, 1467)
#     def forward(self, feature2048, label=None, phase='train'):
#         # Slice the input tensor
#         part, global_ = self.slice(feature2048)

#         # Apply the local fully connected layer
#         simpleconnection = self.simpleconnection(part.view(part.size(0), -1))

#         # Apply Tanh activation
#         tanhout = self.tanhout(simpleconnection)

#         # Reshape the global tensor
#         reglobal = self.reglobal(global_)

#         # Concatenate tanhout and reglobal
#         data2048 = self.concat(reglobal, tanhout)

#         # Fully Connected layer (ip2)
#         ip2 = self.ip2(data2048)

#         if phase == 'train':
#             # Compute softmax and cross-entropy loss
#             loss_final = F.cross_entropy(ip2, label)
#             return loss_final
#         elif phase == 'test':
#             # Compute accuracy during testing
#             accuracy_final = torch.mean((torch.argmax(ip2, dim=1) == label).float())
#             accuracy_top5_final = torch.mean((torch.topk(ip2, k=5)[1].transpose(0, 1) == label.unsqueeze(1)).any(dim=1).float())
#             return accuracy_final, accuracy_top5_final
        
# class FEN(nn.Module):
#     def __init__(self):
#         super(FEN, self).__init__()
#         self.pool = nn.AvgPool2d(4, 4)
#         self.conv1 = nn.Conv2d(3, 20, kernel_size= 5, stride= 1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.ReLU = nn.ReLU()
#         self.conv2 = nn.Conv2d(20, 20, kernel_size= 5, stride= 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(20*1*13, 20)
#         self.fc2 =  nn.Linear(20, 6)

#         self.sampler = SpatialTransformer()
    
#     def forward(self, x):
#         y = self.pool(x)
#         y = self.conv1(y)
#         y = self.pool1(y)
#         y = self.ReLU(y)
#         y = self.conv2(y)
#         y = self.pool2(y)
#         y = self.ReLU(y)
#         y = y.view(y.size(0), -1)
#         y = self.fc1(y)
#         y = self.ReLU(y)
#         y = self.fc2(y)   
#         y = y.view(-1, 2, 3)
#         out = self.sampler(x, y)
#         return out

# class FEN_body(nn.Module):
#     #256x128
#     def __init__(self):
#         super(FEN_body, self).__init__()
#         self.pool = nn.AvgPool2d(4, 4)
#         self.conv1 = nn.Conv2d(3, 20, kernel_size= 5, stride= 1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.ReLU = nn.ReLU()
#         self.conv2 = nn.Conv2d(20, 20, kernel_size= 5, stride= 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(20*5*13, 20)
#         self.fc2 =  nn.Linear(20, 6)

#         self.sampler = SpatialTransformer()


#     def forward(self, x):
#         y = self.pool(x)
#         y = self.conv1(y)
#         y = self.pool1(y)
#         y = self.ReLU(y)
#         y = self.conv2(y)
#         y = self.pool2(y)
#         y = self.ReLU(y)
#         print(y.shape)
#         y = y.view(y.size(0), -1)
#         y = self.fc1(y)
#         y = self.ReLU(y)
#         y = self.fc2(y)   
#         y = y.view(-1, 2, 3)
#         out = self.sampler(x, y)
#         return out

# class MODEL(nn.Module):
#     def __init__(self, num_classes, loss = 'softmax'):
#         super(MODEL, self).__init__()
#         self.fen = FEN()
#         self.fen_body = FEN_body()
#         self.gap = nn.AdaptiveAvgPool2d(1024, 1)
#         self.linear = nn.Linear(1024, 1024)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(2048, num_classes)

#     def forward(self, img):
#         crop_head_img, crop_larm_img, crop_rarm_img, crop_body_img, crop_lleg_img, crop_rleg_img = pose(img)  
#         l_arm = fen(crop_larm_img)
#         r_arm = fen(crop_rarm_img)
#         l_leg = fen(crop_lleg_img)
#         r_leg = fen(crop_rleg_img)
#         body = fen_body(crop_body_img)
#         y1 = torch.cat((l_arm,l_leg), axis = 2)
#         y2 = torch.cat((r_arm,r_leg), axis = 2)
#         black_area = torch.zeros((1, 3, 128, 128))
#         y3 = torch.cat((crop_head_img, body, black_area), axis = 2)
#         y4 = torch.cat((y1, y3, y2), axis = 3)
#         part = self.gap(part)
#         part = self.linear(part)
#         part = self.tanh(part)
#         res = torch.cat((full, part), dim= 1)
#         res = self.classifier(res)
#         return res

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# fen_body = FEN_body()
# fen_body = fen_body.to(device)
# fen = FEN()
# fen = fen.to(device)
# model = MODEL()
# model = model.to(device)
# img = cv2.imread('D:\Phong\TT\market1501\Market-1501-v15.09.15\gt_bbox\\0001_c1s1_002301_00.jpg')
# img = img.transpose(2, 0, 1)
# img = np.expand_dims(img, axis= 0)
# print(img.shape)
# img = torch.from_numpy(img)
# img = img.type(torch.FloatTensor)
# out = model(img)
# print(out)
# out = out.cpu().detach().numpy()
# out = out.squeeze()
# out = out.transpose(1, 2, 0)
# print(out.shape)
# cv2.imwrite('test.jpg',out)