
def move_to(obj):
  """ 
    Recursively move elements within a dictionary or NumPy array to PyTorch tensors.
    
    Args:
        obj: A dictionary or NumPy array containing elements to be converted to PyTorch tensors.
        
    Returns:
        dict or torch.Tensor: The input object with elements converted to PyTorch tensors.
    
    Raises:
        TypeError: If the input type is not a dictionary or NumPy array.
    """
  if isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v)
    return res
  elif isinstance(obj, np.ndarray):
    return torch.from_numpy(obj)
  else:
    raise TypeError("Invalid type for move_to")

def model_select(i : int):
    """ 
    Select and save a PyTorch model based on the given index.
    
    Args:
        i (int): An integer representing the model selection index (1, 2, or 3).
        
    Returns:
        nn.Module: The selected PyTorch model.
    
    Raises:
        ValueError: If an invalid model index is provided.
    """
    if i==1:
        model = MyModel()
    elif i==2:
        model = MyModel2()
    elif i==3:
        model = MyModel3()
    else:
        print("Invalid model")
    saved_model = torch.jit.script(model)
    saved_model.save('saved_model.pt')
    return model

class CustomImageDataset(Dataset):
    """ 
    Custom PyTorch dataset for triplet data.
    
    This dataset loads triplet samples (anchor, positive, negative) for triplet loss training.
    
    Args:
        annotations_file: A pandas DataFrame containing dataset annotations.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A triplet of tensors (anchor, positive, negative).
    """
    def __init__(self, annotations_file):
        self.data = annotations_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx, 0]
        if idx<2872:
            pos = dataset.iloc[0:2871,0].sample().iloc[0]
            neg = dataset.iloc[2872:,0].sample().iloc[0]
        elif idx<3409:
            pos = dataset.iloc[2872:3408,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:2871,0] ,dataset.iloc[3409:,0]]).sample().iloc[0]
        elif idx<12920:
            pos = dataset.iloc[3409:12919,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:3408,0] ,dataset.iloc[12920:,0]]).sample().iloc[0]
        elif idx<17564:
            pos = dataset.iloc[12920:17563,0].sample().iloc[0]
            neg = pd.concat([dataset.iloc[0:12919,0] ,dataset.iloc[17564:,0]]).sample().iloc[0]
        elif idx<=25830:
            pos = dataset.iloc[17564:25830,0].sample().iloc[0]
            neg = dataset.iloc[0:17563,0].sample().iloc[0]
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_anc = transforms.functional.resize(input_batch[0],size=[224, 224])
        input = pos
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_pos = transforms.functional.resize(input_batch[0],size=[224, 224])
        input = neg
        input_image = Image.open(input).convert("RGB")
        input_tensor = convert_to_tensor(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch_neg = transforms.functional.resize(input_batch[0],size=[224, 224])
        return input_batch_anc,input_batch_pos,input_batch_neg


def custom_euc_dis(y_predictions, target):
    """ 
    Compute the custom Euclidean distance loss between predictions and targets.
    
    Args:
        y_predictions: Predicted values.
        target: Target values.
        
    Returns:
        torch.Tensor: The computed custom Euclidean distance loss value.
    """
    loss_value = torch.square(y_predictions - target).sum().sqrt()
    return loss_value

def metric(y_predictions, target):
    """ 
    Compute a custom metric based on Euclidean distances between predictions and targets.
    
    Args:
        y_predictions: Predicted values.
        target: Target values.
        
    Returns:
        torch.Tensor: The computed custom metric value.
    """
    wanted_value = torch.square(y_predictions - target).sum(0).sqrt()
    return torch.mean(wanted_value)

class TripletLoss(nn.Module):
    """ 
    Custom triplet loss function for Siamese networks.
    
    Args:
        margin (float): The margin value for triplet loss.
        
    Returns:
        torch.Tensor: The computed triplet loss value.
    """
    def __init__(self, margin=0.9):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
