import torch.nn as nn
import torch.nn.functional as F

class DogBreedCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
        # Atrous (dilated) convolution: Expands the kernel's receptive field (area of input covered) 
        # to capture more context withx increasing kernel size or reducing spatial resolution(HxW of feature map). 
        # Dilation=6 means elements in the 3x3 kernel are spaced 6 pixels apart. Padding=6 ensures 
        # xput dimensions remain consistent with the input.

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.pool7= nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = None
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500,128)
        self.fc4 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300) -> N=batch size
        :return: lower-level feature maps conv4_3 and conv7
        """
        x = F.relu(self.conv1_1(x))  # (N, 64, 300, 300)
        x = F.relu(self.conv1_2(x))  # (N, 64, 300, 300)
        x = self.pool1(x)  # (N, 64, 150, 150)

        x = F.relu(self.conv2_1(x))  # (N, 128, 150, 150)
        x = F.relu(self.conv2_2(x))  # (N, 128, 150, 150)
        x = self.pool2(x)  # (N, 128, 75, 75)

        x = F.relu(self.conv3_1(x))  # (N, 256, 75, 75)
        x = F.relu(self.conv3_2(x))  # (N, 256, 75, 75)
        x = F.relu(self.conv3_3(x))  # (N, 256, 75, 75)
        x = self.pool3(x)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        x = F.relu(self.conv4_1(x))  # (N, 512, 38, 38)
        x = F.relu(self.conv4_2(x))  # (N, 512, 38, 38)
        x = F.relu(self.conv4_3(x))  # (N, 512, 38, 38)
        conv4_3_feats = x  # (N, 512, 38, 38)
        x = self.pool4(x)  # (N, 512, 19, 19)

        x = F.relu(self.conv5_1(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_2(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_3(x))  # (N, 512, 19, 19)
        x = self.pool5(x)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        x = F.relu(self.conv6(x))  # (N, 1024, 19, 19)
        x = self.pool6(x) # (N,1024,10,10) -> ceil=True

        x = F.relu(self.conv7(x))  # (N, 1024, 10, 10)
        x = self.pool7(x) # (N,1024,5,5)

        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(flattened_size, 5000).to(x.device)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 1024*19*19 = 369664)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
