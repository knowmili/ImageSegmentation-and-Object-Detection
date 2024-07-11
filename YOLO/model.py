"""
DISCLAIMER - 

This is an implementation from scratch of YoloV1 model 
using this tutorial - https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO. 
"""
import torch
from torch import nn


"""
Architectural Configuration of YOLOv1

"""
architecture_config = [
    # Tuple : (kernel size, channel output,stride, padding )
    (7, 64, 2, 3),
    "M", # Maxpool layer
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List : (tuples and then last integer represent the number of repeats of the layers)
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
] # Note - Does not include the Fully connected Layers and only contains the Conv Layers


class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels, **kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self,in_channels=3, **kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecture_config # Architecture of YOLOv1
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) # YOLO Darknet Architecture
        self.fcs = self._create_fcs( **kwargs)

    def forward(self,x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))

    def _create_conv_layers(self,architecture):
        """
        Create the Convolutional Layers of the YOLOv1 model
        from the given architectural config of YOLOv1
        Args: 
            architecture : List of Tuples and List
        Returns:
            nn.Sequential : Sequential Layer of the YOLOv1 Conv model

        """
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                """
                    For the Tuple element in the architecture config
                    Tuple : (kernel size, channel output,stride, padding )
                    Adds One Convolutional Layer to the model
                """ 
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                        )
                    ]
                in_channels = x[1]

            elif type(x) == str:
                """
                    For the String element in the architecture config
                    String : "M"
                    Adds Maxpool Layer to the model
                """
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
            elif type(x) == list:
                """
                    For the List element in the architecture config
                    List : [(1, 256, 1, 0), (3, 512, 1, 1), 4] # Example Config
                    Adds Multiple Convolutional Layers to the model
                """
                conv1 = x[0] #Tuple
                conv2 = x[1] #Tuple
                num_repeats = x[2] #Tuple
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], 
                            kernel_size=conv1[0], 
                            stride=conv1[2], 
                            padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1], conv2[1], 
                            kernel_size=conv2[0], 
                            stride=conv2[2], 
                            padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    def _create_fcs(self,split_size,num_boxes,num_classes):
        """
        Create the Fully Connected Layers of the YOLOv1 model
        Args: 
            split_size : Grid Size
            num_boxes : Number of Bounding Boxes
            num_classes : Number of Classes
        Returns:
            nn.Sequential : Sequential Layer of the YOLOv1 FC model
        """
        S,B,C = split_size,num_boxes,num_classes
        return nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1024*S*S,496), # In the original architecture, it is 4096
                                nn.Dropout(0.0),
                                nn.LeakyReLU(0.1),
                                nn.Linear(496,S * S * ( C + B * 5)) # Grid Size * (Classes + Bounding Box) - (S,S,30)
                            )
    
def test(S=7,B=2,C=20):
    model = Yolov1(split_size = S,num_boxes=B,num_classes=C)
    x = torch.randn((2,3,448,448))
    print(model(x).shape) # Expected (N, S * S * (C + B * 5))
    import torchinfo
    print(torchinfo.summary(model, input_size=(1,3,448,448))  )

test()
        

                
                