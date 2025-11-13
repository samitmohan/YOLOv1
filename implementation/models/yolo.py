import torch.nn as nn 
import torchvision

'''
What final layer predicts:
[
    x_offset_box1,y_offset_box1,sqrt_w_box1,sqrt_h_box1,conf_box1, # box-1 params
    ...,
    x_offset_boxB,y_offset_boxB,sqrt_w_boxB,sqrt_h_boxB,conf_boxB, # box-B params
    p1, p2, ...., pC-1, pC  # class conditional probabilities
] for each S*S grid cell
'''
class YOLOV1(nn.Module):
    # Backbone of resnet34 (pretrained on 224*224 img from Imagenet)
    # 4 Conv, BatchNorm, LeakuRELU for Detection Head
    # FC Layer (Final: S*S*(5B+C)) output dim)
    def __init__(self, img_size, num_classes, model_config):
        super(YOLOV1, self).__init__()
        self.img_size = img_size
        self.img_channels = model_config('img_channels')
        self.backbone_channels = model_config['backbone_channels']
        self.yolo_convChannel = model_config(['yolo_convChannel'])
        self.conv_spatial_size = model_config['conv_spatial_size']
        self.leaky_relu_slope = model_config['leaky_relu_slope']
        self.yolo_fc_hidden_dim = model_config['fc_dim']
        self.yolo_fc_dropout_prob = model_config['fc_dropout']
        self.use_conv = model_config['use_conv']
        self.S = model_config['S']
        self.B = model_config['B']
        self.C = num_classes
        backbone = torchvision.models.resnet34( weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # detection
        # 3-> 3x3 conv layers followed by 1 1x1 layer (BatchNorm -> Relu)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.backbone_channels,
                      self.yolo_convChannel,
                      3,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_convChannel),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_convChannel, self.yolo_convChannel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_convChannel),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_convChannel, self.yolo_convChannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.yolo_convChannel),
            nn.LeakyReLU(self.leaky_relu_slope)
        )

        if self.use_conv:
            self.fc_yolo_layers = nn.Sequential(nn.Conv2d(self.yolo_convChannel, 5 * self.B + self.C, 1),)
        else:
            # Final layer
            self.fc_yolo_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.conv_spatial_size * self.conv_spatial_size * self.yolo_convChannel * self.yolo_fc_hidden_dim),
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.yolo_fc_dropout_prob),
                nn.Lienar(self.yolo_fc_hidden_dim, self.S * self.S * (5 * self.B + self.C)),
            )

    def forward(self x):
        out = self.features(x)
        out = self.conv_layers(out)
        out = self.fc_yolo_layers(out)
        if self.use_conv:
            out = out.permute(0,2,3,1) # reshape: batch * s * s * (5b+c)
        return out