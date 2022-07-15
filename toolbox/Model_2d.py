# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import numpy as np
from pprint import pprint

__all__ = [
    "ResidualConvBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]

#............................
#............................
#............................

class ResidualConvBlock(nn.Module):
    """Implements residual conv function.
    Args:
        channels (int): Number of channels in the input image.
    """
#...!...!..................
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rc_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels)
        )
#...!...!..................
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.rc_block(x)
        out = out + identity
        return out

#............................
#............................
#............................
class Discriminator(nn.Module):
#...!...!..................
    def __init__(self,num_inp_chan,num_hr_bins,conf, verb=0) -> None:
        super(Discriminator, self).__init__()
        if verb:
            print('D_Model conf='); pprint(conf)
        self.verb=verb
        reluSlope=0.2
        
        self.inp_shape=(num_inp_chan,num_hr_bins,num_hr_bins)

        # .....  CNN layers
        hpar1=conf['conv_block']
        self.cnn_block = nn.ModuleList()
        inp_chan=num_inp_chan
        for out_chan,cnnker,cnnbn,cnnstr in zip(hpar1['filter'],hpar1['kernel'],hpar1['bn'],hpar1['stride']):
            # nn.Conv2d( in_channels,out_channels,kernel_size,stride=1,padding=0
            #torch.nn.BatchNorm2d(num_features,
            doBias=not cnnbn
            self.cnn_block.append( nn.Conv2d(inp_chan, out_chan, cnnker, stride=cnnstr,padding=1, bias=doBias))
            if cnnbn: self.cnn_block.append( nn.BatchNorm2d(out_chan))
            self.cnn_block.append( nn.LeakyReLU(reluSlope, True))
            inp_chan=out_chan

        ''' Automatically compute the size of the output of CNN+Pool block,  
        needed as input to the first FC layer 
        '''

        with torch.no_grad():
            # process 2 fake examples through the CNN portion of model
            x1=torch.tensor(np.zeros((2,)+self.inp_shape), dtype=torch.float32)
            y1=self.forwardCnnOnly(x1)
            self.flat_dim=np.prod(y1.shape[1:])
            if self.verb>2: print('D-net flat_dim=',self.flat_dim)
        
        # .... add FC  layers
        hpar2=conf['fc_block']
        self.fc_block  = nn.ModuleList()
        inp_dim=self.flat_dim
                
        for i,dim in enumerate(hpar2['dims']):
            self.fc_block.append( nn.Dropout(p= hpar2['dropFrac']))            
            self.fc_block.append( nn.Linear(inp_dim,dim))
            self.fc_block.append( nn.LeakyReLU(reluSlope,True))
            inp_dim=dim

        self.fc_block.append( nn.Linear(inp_dim,1))
        self.fc_block.append(nn.Sigmoid())  # the decision

#...!...!..................
    def forwardCnnOnly(self, x):
        #X flatten 2D image 
        #x=x.view((-1,)+self.inp_shape )

        if self.verb>2: print('J: inp2cnn',x.shape,x.dtype)
        for i,lyr in enumerate(self.cnn_block):
            if self.verb>2: print('Jcnn-lyr: ',i,lyr)
            x=lyr(x)
            if self.verb>2: print('Jcnn: out ',i,x.shape)
        return x
    
#...!...!..................      
    def forward(self, x: Tensor) -> Tensor:
        if self.verb>2: print('DFa:',x.shape,x.dtype)
        x=self.forwardCnnOnly(x)        
        x = torch.flatten(x, 1)
        if self.verb>2: print('DFb:',x.shape)
        for i,lyr in enumerate(self.fc_block):
            if self.verb>2: print('DFc: ',i,lyr.shape)
            x=lyr(x)
            if self.verb>2: print('DFd: ',i,x.shape)
        if self.verb>2: print('DF y',x.shape)        
        return x
    
#...!...!..................
    def short_summary(self):
        numLayer=sum(1 for p in self.parameters())
        numParams=sum(p.numel() for p in self.parameters())
        return {'modelWeightCnt':numParams,'trainedLayerCnt':numLayer,'modelClass':self.__class__.__name__}


#............................
#............................
#............................
class Generator(nn.Module):
    def __init__(self,num_inp_chan,conf, verb=0) -> None:
        super(Generator, self).__init__()
        if verb:
            print('G_Model conf='); pprint(conf)
        self.verb=verb

        # First conv layer.        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_inp_chan, conf['first_cnn_chan'], (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(conf['num_resid_conv']):
            trunk.append(ResidualConvBlock(conf['first_cnn_chan']))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(conf['first_cnn_chan'], conf['first_cnn_chan'], (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d( conf['first_cnn_chan'])
        )

        # Upscale conv block.
        trunk2 = []  ; assert conf['num_upsamp']==2, not_tested
        upsamp_chan=conf['first_cnn_chan']*2*2  # because: PixelShuffle(2)
        # Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape (*, C, H * r, W * r), where r is an upscale factor.

 
        for _ in range(conf['num_upsamp']):
            trunk2.append(nn.Conv2d(conf['first_cnn_chan'], upsamp_chan, (3, 3), (1, 1), (1, 1)))
            trunk2.append(nn.PixelShuffle(2))  #efficient sub-pixel convolution stride1/2
            trunk2.append(nn.PReLU())
            
        self.upsampling = nn.Sequential(*trunk2)

        # Output layer.
        self.conv_block3 =  nn.Sequential(
            nn.Conv2d(conf['first_cnn_chan'], num_inp_chan, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )  # to generate only positive values

        # Initialize neural network weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        #print('gfx0',x.shape)
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        #print('gfx',x.shape,x.dtype)
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        #print('gf2',out1.shape,out2.shape)
        out = out1 + out2
        
        out = self.upsampling(out)
        out = self.conv_block3(out)
        #print('gf3',out.shape)
        return out
#...!...!..................
    def _initialize_weights(self) -> None:
        for m in self.modules():
            #print('Gm',m,isinstance(m, nn.Conv2d))
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1
#...!...!..................
    def short_summary(self):
        numLayer=sum(1 for p in self.parameters())
        numParams=sum(p.numel() for p in self.parameters())
        return {'modelWeightCnt':numParams,'trainedLayerCnt':numLayer,'modelClass':self.__class__.__name__}



#............................
#............................
#............................

class ContentLoss(nn.Module):
    """ Constructs a content loss function based on the VGG19 network.
        Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # Standardized operations.
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        # Find the feature map difference between the two images.
        loss = F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
