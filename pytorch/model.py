import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#more filters mean the model can detect more patterns in the image
#binary classification
class LaneNet(nn.Module): #neural network
    def __init__(self):
        super(LaneNet, self).__init__() #initialize and inherit the necessary functionality from parent class 
        self.dropout = nn.Dropout2d(0.3)
        self.encoder1 = self.encoder_layer(3, 64)
        self.encoder2 = self.encoder_layer(64, 128)
        self.encoder3 = self.encoder_layer(128, 256)
        self.encoder4 = self.encoder_layer(256, 512)
        
        self.decoder4 = self.decoder_layer(512, 256)
        self.decoder3 = self.decoder_layer(512, 128)  # 256 + 256 from skip connection
        self.decoder2 = self.decoder_layer(256, 64)   # 128 + 128 from skip connection
        self.decoder1 = self.decoder_layer(128, 64)   # 64 + 64 from skip connection
        
        self.final = nn.Conv2d(64, 1, kernel_size=1) #segmentation mask

    def encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(   # pass the input through all layers in order and return the final output
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #2d images
            nn.BatchNorm2d(out_channels), #increase speed, helps with overfitting
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # faster execution, rewriting existing image, introduce non linearity after linear transformations
            nn.MaxPool2d(2), #downsampling
            self.dropout
        )

    def decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2), #upsample, output will be twice the size of the input in both height and width
            self.dropout
        )
    
    def forward(self, image): 
        e1 = self.encoder1(image)  #nn module
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Decoder with skip connections
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat([d4, F.interpolate(e3, size=d4.shape[2:])], dim=1))  #combine high-level features  with low-level features 
        d2 = self.decoder2(torch.cat([d3, F.interpolate(e2, size=d3.shape[2:])], dim=1))  #Preserve info from earlier layers
        d1 = self.decoder1(torch.cat([d2, F.interpolate(e1, size=d2.shape[2:])], dim=1))
        
        d1 = F.interpolate(d1, size=(590, 1640), mode='bilinear', align_corners=False)
        return self.final(d1)