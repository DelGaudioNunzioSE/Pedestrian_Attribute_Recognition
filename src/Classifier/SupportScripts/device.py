import torch


def device_selecter():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}") 
    return DEVICE