import torch
import torch.nn as nn
import torchvision.models as models

numOfAU = 17

def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1):
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim()
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.mul(size)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div


class LocalResponseNorm(nn.Module):
    r"""Applies local response normalization over an input signal composed
    of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, ...)`
        - Output: :math:`(N, C, ...)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)


class ResNet101(nn.Module):
    def __init__(self, Dim):
        super(ResNet101, self).__init__()
        
        self.Dim = Dim

        # Original Model
        self.backbone = models.resnet101(pretrained=False)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=7, bias=True)

        # Universal Module
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=8) # W * H --> W/8 * H/8
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4) # W * H --> W/4 * H/4
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # W * H --> W/2 * H/2
        self.GAP = nn.AdaptiveAvgPool2d((1,1))                # B * C * W * H --> B * C * 1 * 1
        
        # Expression Recognition
        self.LRN_em = LocalResponseNorm(2)
        self.reduce_dim_em = nn.Sequential(
            nn.Conv2d(in_channels=3840,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        ) 
        self.pred_em = nn.Linear(in_features=1024, out_features=7, bias=True)

        # AU Recognition
        self.LRN_au = LocalResponseNorm(2)
        self.pred_au = nn.Sequential(nn.Linear(256 * numOfAU * 2, numOfAU), nn.Sigmoid())
        
        # Deconvolution Layer
        # ConvTranspose2d: output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Reduce Dimension
        self.reduce_dim_1_au = nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.reduce_dim_2_au = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.reduce_dim_3_au = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ) 
        # Crop Net
        self.Crop_Net_1 = nn.ModuleList([ nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()) for i in range(numOfAU * 2) ])
        self.Crop_Net_2 = nn.ModuleList([ nn.Sequential( nn.Linear(in_features=64*24*24, out_features=256), nn.ReLU(), nn.Dropout(p=0.5) ) for i in range(numOfAU * 2) ])

        # Bilinear + Attention
        self.fc_em_fuse_3 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1024, self.Dim), nn.BatchNorm1d(self.Dim), nn.Tanh())
        self.fc_au_fuse_3 = nn.ModuleList([ nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, self.Dim), nn.BatchNorm1d(self.Dim), nn.Tanh()) for i in range(numOfAU) ])

        # self.fc_attention_fuse_3 = nn.Sequential(nn.Linear(self.Dim,1),) # Share Weight, no ReLU 
        self.fc_attention_fuse_3 = nn.Sequential(nn.Linear(self.Dim, 1), nn.ReLU(inplace=True)) # Share Weight, ReLU

        self.pred_em_fuse_3 = nn.Linear(1024 + 512, 7)
    
        # Generate AU Label
        self.InitTable = torch.Tensor([
            [1.0, 1.0, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.5, 0.1, 0.5, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.5, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0, 0.5, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
            [1.0, 0.1, 0.5, 0.1, 0.5, 1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 1.0, 1.0, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 1.0, 1.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]).float()
        self.PriorKnowledgeTable = nn.Parameter(self.InitTable)

        # Generate AU Mask
        self.Mask_A = torch.Tensor([
             [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
             [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).float().cuda()
        self.Mask_B = torch.Tensor([
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).float().cuda()
        self.Mask_C = torch.Tensor([
             [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
             [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).float().cuda()       

    def forward(self, input, args): # input: 3 * 224 * 224

        if args.Experiment == 'EM':
            featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input)))) # 64 * 56 * 56
            featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
            featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
            featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
            featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

            featureMap1 = self.maxpool1(featureMap1)                               # 256 * 56 * 56 --> 256 * 7 * 7
            featureMap2 = self.maxpool2(featureMap2)                               # 512 * 28 * 28 --> 512 * 7 * 7
            featureMap3 = self.maxpool3(featureMap3)                               # 1024 * 14 * 14 --> 1024 * 7 * 7
            featureMap = torch.cat((torch.cat((featureMap1, featureMap2), dim=1), torch.cat((featureMap3, featureMap4), dim=1)), dim=1) # 3840 * 7 * 7

            # Save GPU Memory
            del featureMap0, featureMap1, featureMap2, featureMap3, featureMap4
            torch.cuda.empty_cache()
            
            featureMap = self.LRN_em(featureMap)                                   # 3840 * 7 * 7 --> 3840 * 7 * 7
            featureMap = self.reduce_dim_em(featureMap)                            # 3840 * 7 * 7 --> 1024 * 7 * 7
            
            feature = self.GAP(featureMap)                                         # 1024 * 7 * 7 --> 1024 * 1 * 1

            # Save GPU Memory
            del featureMap
            torch.cuda.empty_cache()

            feature = feature.view(feature.size(0),feature.size(1))                # 1024 * 1 * 1 --> 1024
            pred = self.pred_em(feature)                                           # 1024 --> 7

            return pred

        elif args.Experiment == 'AU':
            
            input, au_loc = input # input = (input, au_loc)

            with torch.no_grad():
                featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input)))) # 64 * 56 * 56
                featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
                featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
                featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
                featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

            deconv_featureMap3 = self.deconv_layer1(featureMap4)                       # 2048 * 7 * 7 --> 1024 * 14 * 14
            deconv_featureMap3 = torch.cat((featureMap3, deconv_featureMap3), dim=1)   # cat(1024 * 14 * 14, 1024 * 14 * 14) = 2048 * 14 * 14
            deconv_featureMap3 = self.reduce_dim_1_au(deconv_featureMap3)              # 2048 * 14 * 14 --> 1024 * 14 * 14
            
            deconv_featureMap2 = self.deconv_layer2(deconv_featureMap3)                # 1024 * 14 * 14 --> 512 * 28 * 28
            deconv_featureMap2 = torch.cat((featureMap2, deconv_featureMap2), dim=1)   # cat(512 * 28 * 28, 512 * 28 * 28) = 1024 * 28 * 28
            deconv_featureMap2 = self.reduce_dim_2_au(deconv_featureMap2)              # 1024 * 28 * 28 --> 512 * 28 * 28

            deconv_featureMap1 = self.deconv_layer3(deconv_featureMap2)                # 512 * 28 * 28 --> 256 * 56 * 56
            deconv_featureMap1 = torch.cat((featureMap1, deconv_featureMap1), dim=1)   # cat(256 * 56 * 56, 256 * 56 * 56) = 512 * 56 * 56
            deconv_featureMap1 = self.reduce_dim_3_au(deconv_featureMap1)              # 512 * 56 * 56 --> 64 * 56 * 56
            
            deconv_featureMap = self.LRN_au(deconv_featureMap1)                        # 64 * 56 * 56 --> 64 * 56 * 56
            
            au_featureMap = self.crop_au_featureMap(deconv_featureMap, au_loc)         # crop au feature: (2 * numOfAU) * 256
            au_featureMap = au_featureMap.view(au_featureMap.size(0), -1)              # (2 * numOfAU) * 256 --> (2 * numOfAU) * 256
            pred = self.pred_au(au_featureMap)                                         # (2 * numOfAU) * 256 --> (2 * numOfAU)

            return pred

        elif args.Experiment == 'Fuse':

            input, au_loc = input # input = (input, au_loc)

            with torch.no_grad():
                # Feature Map
                featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input)))) # 64 * 56 * 56
                featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
                featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
                featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
                featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

                # AU
                deconv_featureMap3 = self.deconv_layer1(featureMap4)                   # 2048 * 7 * 7 --> 1024 * 14 * 14
                deconv_featureMap3 = torch.cat((featureMap3,deconv_featureMap3),dim=1) # cat(1024 * 14 * 14, 1024 * 14 * 14) = 2048 * 14 * 14
                deconv_featureMap3 = self.reduce_dim_1_au(deconv_featureMap3)          # 2048 * 14 * 14 --> 1024 * 14 * 14
                
                deconv_featureMap2 = self.deconv_layer2(deconv_featureMap3)            # 1024 * 14 * 14 --> 512 * 28 * 28
                deconv_featureMap2 = torch.cat((featureMap2,deconv_featureMap2),dim=1) # cat(512 * 28 * 28, 512 * 28 * 28) = 1024 * 28 * 28
                deconv_featureMap2 = self.reduce_dim_2_au(deconv_featureMap2)          # 1024 * 28 * 28 --> 512 * 28 * 28

                deconv_featureMap1 = self.deconv_layer3(deconv_featureMap2)            # 512 * 28 * 28 --> 256 * 56 * 56
                deconv_featureMap1 = torch.cat((featureMap1,deconv_featureMap1),dim=1) # cat(256 * 56 * 56, 256 * 56 * 56) = 512 * 56 * 56
                deconv_featureMap1 = self.reduce_dim_3_au(deconv_featureMap1)          # 512 * 56 * 56 --> 64 * 56 * 56
                
                deconv_featureMap = self.LRN_au(deconv_featureMap1)                    # 64 * 56 * 56 --> 64 * 56 * 56
                
                au_featureMap = self.crop_au_featureMap(deconv_featureMap,au_loc)      # crop au feature: (2*numOfAU) * 256
                
                # EM
                featureMap1 = self.maxpool1(featureMap1)                               # 256 * 56 * 56 --> 256 * 7 * 7
                featureMap2 = self.maxpool2(featureMap2)                               # 512 * 28 * 28 --> 512 * 7 * 7
                featureMap3 = self.maxpool3(featureMap3)                               # 1024 * 14 * 14 --> 1024 * 7 * 7
                featureMap = torch.cat((torch.cat((featureMap1,featureMap2),dim=1),torch.cat((featureMap3,featureMap4),dim=1)),dim=1) # 3840 * 7 * 7
                
                featureMap = self.LRN_em(featureMap)                                   # 3840 * 7 * 7 --> 3840 * 7 * 7
                featureMap = self.reduce_dim_em(featureMap)                            # 3840 * 7 * 7 --> 1024 * 7 * 7
                feature = self.GAP(featureMap)                                         # 1024 * 7 * 7 --> 1024 * 1 * 1
                feature = feature.view(feature.size(0),feature.size(1))                # 1024 * 1 * 1 --> 1024

            # MultiScale
            pred1 = self.pred_em(feature)

            # Bilinear
            # AU_feature = torch.cat((au_featureMap[:,0:12,:],au_featureMap[:,12:,:]),dim=2)
            AU_feature = au_featureMap.view(au_featureMap.size(0), 2, numOfAU, au_featureMap.size(2)).transpose(1, 2).contiguous().view(au_featureMap.size(0), numOfAU, -1) # numOfAU * 512
            AU_Dim_feature = torch.zeros((AU_feature.size(0), numOfAU, self.Dim)).cuda() # numOfAU * Dim
            for i in range(numOfAU):
                AU_Dim_feature[:, i, :] = self.fc_au_fuse_3[i](AU_feature[:, i, :])

            EM_Dim_feature = self.fc_em_fuse_3(feature) # Dim
            EM_Dim_feature = EM_Dim_feature.view(EM_Dim_feature.size(0), 1, self.Dim).repeat(1, numOfAU, 1) # numOfAU * Dim

            Attention = EM_Dim_feature * AU_Dim_feature # numOfAU * Dim

            Attention_Result = torch.zeros((Attention.size(0), numOfAU, 1)).cuda() # numOfAU * 1
            for i in range(numOfAU):
                Attention_Result[:, i, :] = self.fc_attention_fuse_3(Attention[:, i, :]) # Share Weight FC

            # Attention_Result = nn.Sigmoid()(Attention_Result) # Sigmoid
            Attention_Result = nn.Softmax(dim=1)(Attention_Result) # Softmax

            au_prob = Attention_Result.view(Attention_Result.size(0), numOfAU)

            Attention_Result = Attention_Result.repeat(1, 1, 512) # numOfAU * 512

            Result = Attention_Result * AU_feature # numOfAU * 512

            Result = Result.sum(dim=1)             # 512

            Result = torch.cat((feature, Result),dim=1)

            # Bilinear Pooling
            pred2 = self.pred_em_fuse_3(Result)

            return pred1, pred2, au_prob

    def crop_au_featureMap(self, deconv_featureMap, au_location):
        au = []
        for i in range(numOfAU):
            au.append(au_location[:,i,:])

        batch_size = deconv_featureMap.size(0)
        map_ch = deconv_featureMap.size(1)
        map_len = deconv_featureMap.size(2)

        grid_ch = map_ch
        grid_len = int(map_len * 24 / 56)

        feature_list = []
        for i in range(numOfAU):
            grid1_list = []
            grid2_list = []
            for j in range(batch_size):
                h_min_1 = au[i][j,1]-int(grid_len/2)
                h_max_1 = au[i][j,1]+int(grid_len/2)
                w_min_1 = au[i][j,0]-int(grid_len/2)
                w_max_1 = au[i][j,0]+int(grid_len/2)
             
                h_min_2 = au[i][j,3]-int(grid_len/2)
                h_max_2 = au[i][j,3]+int(grid_len/2)
                w_min_2 = au[i][j,2]-int(grid_len/2)
                w_max_2 = au[i][j,2]+int(grid_len/2)
                # grid_1 = deconv_featureMap[j, :, h_min_1:h_max_1, w_min_1:w_max_1]
                # grid_2 = deconv_featureMap[j, :, h_min_2:h_max_2, w_min_2:w_max_2]

                map_h_min_1 = max(0, h_min_1)
                map_h_max_1 = min(map_len, h_max_1)
                map_w_min_1 = max(0, w_min_1)
                map_w_max_1 = min(map_len, w_max_1)

                map_h_min_2 = max(0, h_min_2)
                map_h_max_2 = min(map_len, h_max_2)
                map_w_min_2 = max(0, w_min_2)
                map_w_max_2 = min(map_len, w_max_2)
             
                grid_h_min_1 = max(0, 0-h_min_1)
                grid_h_max_1 = grid_len + min(0, map_len-h_max_1)
                grid_w_min_1 = max(0, 0-w_min_1)
                grid_w_max_1 = grid_len + min(0, map_len-w_max_1)

                grid_h_min_2 = max(0, 0-h_min_2)
                grid_h_max_2 = grid_len + min(0, map_len-h_max_2)
                grid_w_min_2 = max(0, 0-w_min_2)
                grid_w_max_2 = grid_len + min(0, map_len-w_max_2)

                grid_1 = torch.zeros(grid_ch, grid_len, grid_len)
                grid_2 = torch.zeros(grid_ch, grid_len, grid_len)
                grid_1 = grid_1.cuda()
                grid_2 = grid_2.cuda()

                grid_1[:, grid_h_min_1:grid_h_max_1, grid_w_min_1:grid_w_max_1] = deconv_featureMap[j, :, map_h_min_1:map_h_max_1, map_w_min_1:map_w_max_1] 
                grid_2[:, grid_h_min_2:grid_h_max_2, grid_w_min_2:grid_w_max_2] = deconv_featureMap[j, :, map_h_min_2:map_h_max_2, map_w_min_2:map_w_max_2] 

                grid1_list.append(grid_1)
                grid2_list.append(grid_2)

            input1 = torch.stack(grid1_list, dim=0)
            input2 = torch.stack(grid2_list, dim=0)
            feature_list.append(input1)
            feature_list.append(input2)
 
        # feature list: (numOfAU * 2) * batch * 1024 * 3 * 3
        output_list = []
        # Feed into crop net individually
        for i in range(numOfAU * 2):
            output = self.Crop_Net_1[i](feature_list[i])
            # output = self.GAP(output)
            output = output.view(batch_size, -1)
            output = self.Crop_Net_2[i](output)
            output_list.append(output)

        au_feature = torch.stack(output_list, dim=1) # batch * (2 * numOfAU) * 256
        # au_feature = torch.cat(output_list, 1) #  batch * (12 * 2 * 150)
        # au_feature = au_feature.view(batch_size, -1, self.feature_dim_au) 

        return au_feature

    def get_au_target(self, EM_Target):

        # get one-hot label
        EM_Target_One_Hot = torch.zeros(EM_Target.size(0), 7).float()
        EM_Target_One_Hot.scatter_(1, EM_Target.view(-1, 1).long(), 1.)
        EM_Target_One_Hot = EM_Target_One_Hot.cuda()

        # clamp
        Clamp_Matrix_A = torch.clamp(self.PriorKnowledgeTable, min=0.75, max=1.00)
        Clamp_Matrix_B = torch.clamp(self.PriorKnowledgeTable, min=0.50, max=0.75)
        Clamp_Matrix_C = torch.clamp(self.PriorKnowledgeTable, min=0.00, max=0.25)
        
        Clamp_Matrix_A = Clamp_Matrix_A * self.Mask_A
        Clamp_Matrix_B = Clamp_Matrix_B * self.Mask_B
        Clamp_Matrix_C = Clamp_Matrix_C * self.Mask_C

        Clamp_Matrix = Clamp_Matrix_A + Clamp_Matrix_B + Clamp_Matrix_C

        # get au target
        AU_Target = EM_Target_One_Hot.mm(Clamp_Matrix)
        # AU_Target = EM_Target_One_Hot.mm(self.PriorKnowledgeTable)

        return AU_Target


class ResNet101_Compound(nn.Module):
    def __init__(self,Dim):
        super(ResNet101_Compound, self).__init__()
        
        self.Dim = Dim

        # Original Model
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=11, bias=True)

        # Universal Module
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=8) # W * H --> W/8 * H/8
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4) # W * H --> W/4 * H/4
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # W * H --> W/2 * H/2
        self.GAP = nn.AdaptiveAvgPool2d((1,1))                # B * C * W * H --> B * C * 1 * 1
        
        # Expression Recognition
        self.LRN_em = LocalResponseNorm(2)
        self.reduce_dim_em = nn.Sequential(
            nn.Conv2d(in_channels=3840, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        ) 
        self.pred_em = nn.Linear(in_features=1024, out_features=11, bias=True)

        # AU Recognition
        self.LRN_au = LocalResponseNorm(2)
        self.pred_au = nn.Sequential(nn.Linear(256 * numOfAU * 2, numOfAU), nn.Sigmoid())
        self.pred_au_noCropNet2 = nn.Sequential(nn.Linear(64 * numOfAU * 2, numOfAU), nn.Sigmoid())
        # ConvTranspose2d: output = (input - 1) * stride + output_padding - 2 * padding + kernel_size
        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Reduce Dimension
        self.reduce_dim_1_au = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.reduce_dim_2_au = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.reduce_dim_3_au = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ) 
        # Crop Net
        self.Crop_Net_1 = nn.ModuleList([ nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU() ) for i in range(numOfAU * 2) ])
        self.Crop_Net_2 = nn.ModuleList([ nn.Sequential( nn.Linear(in_features=64*24*24, out_features=256), nn.ReLU(), nn.Dropout(p=0.5) ) for i in range(numOfAU * 2) ])

        # Feature Fuse

        # Plan 3 : Attention
        # Dropout = 0.5
        self.fc_em_fuse_3 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1024, self.Dim), nn.BatchNorm1d(self.Dim), nn.Tanh())
        self.fc_au_fuse_3 = nn.ModuleList([ nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512,self.Dim), nn.BatchNorm1d(self.Dim), nn.Tanh()) for i in range(numOfAU) ]) # CropNet2
        # self.fc_au_fuse_3 = nn.ModuleList([ nn.Sequential(nn.Dropout(p=0.5),nn.Linear(128,self.Dim),nn.BatchNorm1d(self.Dim),nn.Tanh(),) for i in range(numOfAU) ]) # no CropNet2

        # self.fc_attention_fuse_3 = nn.Sequential(nn.Linear(self.Dim,1),) # Share Weight, no ReLU
        # self.fc_attention_fuse_3 = nn.ModuleList([ nn.Sequential(nn.Linear(self.Dim,1),) for i in range(numOfAU) ]) # no Share Weight, no ReLU
        self.fc_attention_fuse_3 = nn.Sequential(nn.Linear(self.Dim, 1), nn.ReLU(inplace=True)) # Share Weight, ReLU
        # self.fc_attention_fuse_3 = nn.ModuleList([ nn.Sequential(nn.Linear(self.Dim,1),nn.ReLU(inplace=True),) for i in range(numOfAU) ]) # no Share Weight, ReLU

        self.pred_em_fuse_3 = nn.Linear(1024 + 512, 11) # CropNet2
        # self.pred_em_fuse_3 = nn.Linear(1024+128, 11) # no CropNet2

        self.pred_em_fuse_3_attention = nn.Linear(numOfAU * self.Dim, 11)

        self.pred_au_fuse_3 = nn.Sequential(nn.Linear(1024 + 512, numOfAU), nn.Sigmoid())
        self.pred_au_fuse_3_attention = nn.Sequential(nn.Linear(numOfAU * self.Dim, numOfAU), nn.Sigmoid())

        self.pred_em_fuse_3_ShareWeight = nn.Linear(numOfAU * 512, 11) # CropNet2
        # self.pred_em_fuse_3_ShareWeight = nn.Linear(numOfAU * 128, 11) # no CropNet2

    def forward(self, input, args):

        if args.Experiment == 'EM':

            featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input))))
            featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
            featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
            featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
            featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

            featureMap1 = self.maxpool1(featureMap1)                               # 256 * 56 * 56 --> 256 * 7 * 7
            featureMap2 = self.maxpool2(featureMap2)                               # 512 * 28 * 28 --> 512 * 7 * 7
            featureMap3 = self.maxpool3(featureMap3)                               # 1024 * 14 * 14 --> 1024 * 7 * 7
            featureMap = torch.cat((torch.cat((featureMap1,featureMap2),dim=1),torch.cat((featureMap3,featureMap4),dim=1)),dim=1) # 3840 * 7 * 7
            
            featureMap = self.LRN_em(featureMap)                                   # 3840 * 7 * 7 --> 3840 * 7 * 7
            featureMap = self.reduce_dim_em(featureMap)                            # 3840 * 7 * 7 --> 1024 * 7 * 7
            feature = self.GAP(featureMap)                                         # 1024 * 7 * 7 --> 1024 * 1 * 1
            feature = feature.view(feature.size(0),feature.size(1))                # 1024 * 1 * 1 --> 1024
            pred = self.pred_em(feature)                                           # 1024 --> 11

            return pred

        elif args.Experiment == 'AU':
            
            input, au_loc = input # input = (input, au_loc)

            with torch.no_grad():
	            featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input))))
	            featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
	            featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
	            featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
	            featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

          
            deconv_featureMap3 = self.deconv_layer1(featureMap4)                   # 2048 * 7 * 7 --> 1024 * 14 * 14
            deconv_featureMap3 = torch.cat((featureMap3,deconv_featureMap3),dim=1) # cat(1024 * 14 * 14, 1024 * 14 * 14) = 2048 * 14 * 14
            deconv_featureMap3 = self.reduce_dim_1_au(deconv_featureMap3)          # 2048 * 14 * 14 --> 1024 * 14 * 14
            
            deconv_featureMap2 = self.deconv_layer2(deconv_featureMap3)            # 1024 * 14 * 14 --> 512 * 28 * 28
            deconv_featureMap2 = torch.cat((featureMap2,deconv_featureMap2),dim=1) # cat(512 * 28 * 28, 512 * 28 * 28) = 1024 * 28 * 28
            deconv_featureMap2 = self.reduce_dim_2_au(deconv_featureMap2)          # 1024 * 28 * 28 --> 512 * 28 * 28

            deconv_featureMap1 = self.deconv_layer3(deconv_featureMap2)            # 512 * 28 * 28 --> 256 * 56 * 56
            deconv_featureMap1 = torch.cat((featureMap1,deconv_featureMap1),dim=1) # cat(256 * 56 * 56, 256 * 56 * 56) = 512 * 56 * 56
            deconv_featureMap1 = self.reduce_dim_3_au(deconv_featureMap1)          # 512 * 56 * 56 --> 64 * 56 * 56
            
            deconv_featureMap = self.LRN_au(deconv_featureMap1)                    # 64 * 56 * 56 --> 64 * 56 * 56
            
            au_featureMap = self.crop_au_featureMap(deconv_featureMap,au_loc)      # crop au feature: (2*numOfAU) * 256
            au_featureMap = au_featureMap.view(au_featureMap.size(0), -1)          # (2*numOfAU) * 256 --> (2*numOfAU) * 256
            pred = self.pred_au(au_featureMap)                                     # (2*numOfAU) * 256 --> numOfAU
            # pred = self.pred_au_noCropNet2(au_featureMap)                          # no CropNet2
            return pred

        elif args.Experiment == 'Fuse':
            
            input, au_loc = input # input = (input, au_loc)

            with torch.no_grad():
	            featureMap0 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input))))
	            featureMap1 = self.backbone.layer1(featureMap0)                        # 256 * 56 * 56
	            featureMap2 = self.backbone.layer2(featureMap1)                        # 512 * 28 * 28
	            featureMap3 = self.backbone.layer3(featureMap2)                        # 1024 * 14 * 14
	            featureMap4 = self.backbone.layer4(featureMap3)                        # 2048 * 7 * 7

	            # AU
	            deconv_featureMap3 = self.deconv_layer1(featureMap4)                   # 2048 * 7 * 7 --> 1024 * 14 * 14
	            deconv_featureMap3 = torch.cat((featureMap3,deconv_featureMap3),dim=1) # cat(1024 * 14 * 14, 1024 * 14 * 14) = 2048 * 14 * 14
	            deconv_featureMap3 = self.reduce_dim_1_au(deconv_featureMap3)          # 2048 * 14 * 14 --> 1024 * 14 * 14
	            
	            deconv_featureMap2 = self.deconv_layer2(deconv_featureMap3)            # 1024 * 14 * 14 --> 512 * 28 * 28
	            deconv_featureMap2 = torch.cat((featureMap2,deconv_featureMap2),dim=1) # cat(512 * 28 * 28, 512 * 28 * 28) = 1024 * 28 * 28
	            deconv_featureMap2 = self.reduce_dim_2_au(deconv_featureMap2)          # 1024 * 28 * 28 --> 512 * 28 * 28

	            deconv_featureMap1 = self.deconv_layer3(deconv_featureMap2)            # 512 * 28 * 28 --> 256 * 56 * 56
	            deconv_featureMap1 = torch.cat((featureMap1,deconv_featureMap1),dim=1) # cat(256 * 56 * 56, 256 * 56 * 56) = 512 * 56 * 56
	            deconv_featureMap1 = self.reduce_dim_3_au(deconv_featureMap1)          # 512 * 56 * 56 --> 64 * 56 * 56
	            
	            deconv_featureMap = self.LRN_au(deconv_featureMap1)                    # 64 * 56 * 56 --> 64 * 56 * 56
	            
	            au_featureMap = self.crop_au_featureMap(deconv_featureMap,au_loc)      # crop au feature: (2*numOfAU) * 256
	            
	            # EM
	            featureMap1 = self.maxpool1(featureMap1)                               # 256 * 56 * 56 --> 256 * 7 * 7
	            featureMap2 = self.maxpool2(featureMap2)                               # 512 * 28 * 28 --> 512 * 7 * 7
	            featureMap3 = self.maxpool3(featureMap3)                               # 1024 * 14 * 14 --> 1024 * 7 * 7
	            featureMap = torch.cat((torch.cat((featureMap1,featureMap2),dim=1),torch.cat((featureMap3,featureMap4),dim=1)),dim=1) # 3840 * 7 * 7
	            
	            featureMap = self.LRN_em(featureMap)                                   # 3840 * 7 * 7 --> 3840 * 7 * 7
	            featureMap = self.reduce_dim_em(featureMap)                            # 3840 * 7 * 7 --> 1024 * 7 * 7
	            feature = self.GAP(featureMap)                                         # 1024 * 7 * 7 --> 1024 * 1 * 1
	            feature = feature.view(feature.size(0),feature.size(1))                # 1024 * 1 * 1 --> 1024
                 
            # Feature Fuse
            pred1 = self.pred_em(feature)

            # Plan 3
            # AU_feature = torch.cat((au_featureMap[:,0:12,:],au_featureMap[:,12:,:]),dim=2)
            AU_feature = au_featureMap.view(au_featureMap.size(0),2,numOfAU,au_featureMap.size(2)).transpose(1,2).contiguous().view(au_featureMap.size(0),numOfAU,-1) # numOfAU * 512
            AU_Dim_feature = torch.zeros((AU_feature.size(0),numOfAU,self.Dim)).cuda() # numOfAU * Dim
            for i in range(numOfAU):
                AU_Dim_feature[:,i,:] = self.fc_au_fuse_3[i](AU_feature[:,i,:])

            EM_Dim_feature = self.fc_em_fuse_3(feature) # Dim

            EM_Dim_feature = EM_Dim_feature.view(EM_Dim_feature.size(0),1,self.Dim).repeat(1,numOfAU,1) # numOfAU * Dim

            Attention = EM_Dim_feature * AU_Dim_feature # numOfAU * Dim
            
            # Attention: numOfAU * Dim
            # pred2 = self.pred_em_fuse_3_attention(Attention.view(Attention.size(0),-1))
            
            # return pred1, pred2

            Attention_Result = torch.zeros((Attention.size(0),numOfAU,1)).cuda() # numOfAU * 1
            for i in range(numOfAU):
                # Attention_Result[:,i,:] = self.fc_attention_fuse_3[i](Attention[:,i,:]) # No Share Weight FC
                Attention_Result[:,i,:] = self.fc_attention_fuse_3(Attention[:,i,:]) # Share Weight FC

            # Attention_Result = Attention.mean(2,keepdim=True) # GAP

            Attention_Result = nn.Sigmoid()(Attention_Result) # Sigmoid
            # Attention_Result = nn.Softmax(dim=1)(Attention_Result) # Softmax

            au_prob  = Attention_Result.view(Attention_Result.size(0),numOfAU)

            Attention_Result = Attention_Result.repeat(1,1,512) # numOfAU * 512
            # Attention_Result = Attention_Result.repeat(1,1,128) # numOfAU * 128

            Result = Attention_Result * AU_feature # numOfAU * 512

            # 5.22 Concate
            # pred2 = self.pred_em_fuse_3( torch.cat( (feature,Result.mean(dim=1)), dim=1 ) ) 
            # return pred1, pred2, au_prob

            # Share Weight and Feature Concat
            # Result = Result.view(Result.size(0),-1)
            # pred2 = self.pred_em_fuse_3_ShareWeight(Result)

            # return pred1, pred2

            Result = Result.sum(dim=1)             # 512

            Result = torch.cat((feature,Result),dim=1)

            # Bilinear Pooling
            pred2 = self.pred_em_fuse_3(Result)

            return pred1, pred2, au_prob

    def crop_au_featureMap(self,deconv_featureMap,au_location):
        au = []
        for i in range(numOfAU):
            au.append(au_location[:,i,:])

        batch_size = deconv_featureMap.size(0)
        map_ch = deconv_featureMap.size(1)
        map_len = deconv_featureMap.size(2)

        grid_ch = map_ch
        grid_len = int(map_len * 24 / 56)

        feature_list = []
        for i in range(numOfAU):
            grid1_list = []
            grid2_list = []
            for j in range(batch_size):
                h_min_1 = au[i][j,1]-int(grid_len/2)
                h_max_1 = au[i][j,1]+int(grid_len/2)
                w_min_1 = au[i][j,0]-int(grid_len/2)
                w_max_1 = au[i][j,0]+int(grid_len/2)
             
                h_min_2 = au[i][j,3]-int(grid_len/2)
                h_max_2 = au[i][j,3]+int(grid_len/2)
                w_min_2 = au[i][j,2]-int(grid_len/2)
                w_max_2 = au[i][j,2]+int(grid_len/2)
                # grid_1 = deconv_featureMap[j, :, h_min_1:h_max_1, w_min_1:w_max_1]
                # grid_2 = deconv_featureMap[j, :, h_min_2:h_max_2, w_min_2:w_max_2]

                map_h_min_1 = max(0, h_min_1)
                map_h_max_1 = min(map_len, h_max_1)
                map_w_min_1 = max(0, w_min_1)
                map_w_max_1 = min(map_len, w_max_1)

                map_h_min_2 = max(0, h_min_2)
                map_h_max_2 = min(map_len, h_max_2)
                map_w_min_2 = max(0, w_min_2)
                map_w_max_2 = min(map_len, w_max_2)
             
                grid_h_min_1 = max(0, 0-h_min_1)
                grid_h_max_1 = grid_len + min(0, map_len-h_max_1)
                grid_w_min_1 = max(0, 0-w_min_1)
                grid_w_max_1 = grid_len + min(0, map_len-w_max_1)

                grid_h_min_2 = max(0, 0-h_min_2)
                grid_h_max_2 = grid_len + min(0, map_len-h_max_2)
                grid_w_min_2 = max(0, 0-w_min_2)
                grid_w_max_2 = grid_len + min(0, map_len-w_max_2)

                grid_1 = torch.zeros(grid_ch, grid_len, grid_len)
                grid_2 = torch.zeros(grid_ch, grid_len, grid_len)
                grid_1 = grid_1.cuda()
                grid_2 = grid_2.cuda()

                grid_1[:, grid_h_min_1:grid_h_max_1, grid_w_min_1:grid_w_max_1] = deconv_featureMap[j, :, map_h_min_1:map_h_max_1, map_w_min_1:map_w_max_1] 
                grid_2[:, grid_h_min_2:grid_h_max_2, grid_w_min_2:grid_w_max_2] = deconv_featureMap[j, :, map_h_min_2:map_h_max_2, map_w_min_2:map_w_max_2] 

                grid1_list.append(grid_1)
                grid2_list.append(grid_2)

            input1 = torch.stack(grid1_list, dim=0)
            input2 = torch.stack(grid2_list, dim=0)
            feature_list.append(input1)
            feature_list.append(input2)
 
        # feature list: (12 * 2) * batch * 1024 * 3 * 3
        output_list = []
        # Feed into crop net individually
        for i in range(numOfAU * 2):
            output = self.Crop_Net_1[i](feature_list[i])
            # output = self.GAP(output)
            output = output.view(batch_size, -1)
            output = self.Crop_Net_2[i](output)
            output_list.append(output)

        au_feature = torch.stack(output_list, dim=1) # batch * 24 * 256
        # au_feature = torch.cat(output_list, 1) #  batch * (12 * 2 * 150)
        # au_feature = au_feature.view(batch_size, -1, self.feature_dim_au) 

        return au_feature