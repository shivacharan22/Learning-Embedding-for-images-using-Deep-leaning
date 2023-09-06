import torch
import torch.nn as nn

class MyModel3(nn.Module):
    """ 
    Custom PyTorch model for image feature extraction using Swin Transformer.
    
    This model applies Swin Transformer with pretrained weights for image feature extraction.
    It performs image resizing, normalization, and adaptive average pooling to produce a feature tensor.
    
    Args:
        None

    Returns:
        torch.Tensor: A tensor containing extracted image features.

    Example:
        model = MyModel3()
        features = model(input_image)
    """
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
  """ 
    Custom PyTorch model for image feature extraction using DeiT (Data-efficient Image Transformer).
    
    This model applies DeiT with pretrained weights for image feature extraction.
    It performs image resizing, normalization, and adaptive average pooling to produce a feature tensor.
    
    Args:
        None

    Returns:
        torch.Tensor: A tensor containing extracted image features.

    Example:
        model = MyModel2()
        features = model(input_image)
    """
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
    """ 
    Custom PyTorch model for image feature extraction using DeiT (Data-efficient Image Transformer).
    
    This model applies DeiT with pretrained weights for image feature extraction.
    It performs image resizing, normalization, and linear projection to reduce the feature dimensionality.
    
    Args:
        None

    Returns:
        torch.Tensor: A tensor containing extracted image features.

    Example:
        model = MyModel()
        features = model(input_image)
    """
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