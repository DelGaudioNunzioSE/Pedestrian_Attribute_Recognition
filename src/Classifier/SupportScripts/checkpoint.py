
from datetime import datetime

import torch


def checkpoint_fuction(timestamp, model, optimizer, epoch, comment=None):
    if(timestamp == True):
        timestamp = datetime.now().strftime('%d_%H%M')  # Formato: DDMM_HHMMSS
    else:
        timestamp = 'try'

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # save the current epoch
    }
    epoch_and_timestamp = f'{comment}__{epoch}_{timestamp}'
    checkpoint_filename = f'./src/Classifier/Models/{comment}_{epoch}_{timestamp}.pth'
    torch.save(checkpoint, checkpoint_filename)
    print("Model and optimizer saved successfully!")
    return epoch_and_timestamp