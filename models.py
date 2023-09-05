import torch
import torch.nn as nn

class MyModel3(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('swin_large_patch4_window7_224' , pretrained  = True  , num_classes = 0)
        self.layer = nn.AdaptiveAvgPool1d(64)
        self.feature_extractor = model
    def forward(self, x):
        x = transforms.functional.resize(x,size=[224, 224])
        x = x/255.0
        x = transforms.functional.normalize(x, 
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        x = self.layer(self.feature_extractor(x))
        return x

class MyModel2(nn.Module):
  def __init__(self):
    super().__init__()
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()
    model.head = nn.AdaptiveAvgPool1d(64)
    self.feature_extractor = model
  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    x = self.feature_extractor(x)
    
    return x

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()
    model.head = nn.Linear(768,64)
    self.feature_extractor = model
  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    x = self.feature_extractor(x)
    
    return x