import torch.nn as nn 

# Loss Function
def adjustedLoss(prediction, labels, pos_weight ):

    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight) # object to evaluate sigmoid and then LOSS

    loss = criterion(prediction, labels.unsqueeze(1)) # evaluate loss

    mask = labels != -1 #prendo tutti gli indici delle labels -1
    valid_losses = loss[mask] #mi salvo le loss valide, con labels != -1
    mean_loss = valid_losses.mean() #ci faccio la media
    loss[~mask] = mean_loss #la sostituisco al posto delle labels -1

    batch_loss = loss.mean() #ritorno la media con le nuove loss

    return batch_loss


def total_loss_fuction(loss_gender,loss_hat,loss_bag, gender_weight = 1/3,  bag_weight=1/3, hat_weight=1/3):
      if (gender_weight+hat_weight+bag_weight) != 1:
            print('Total weight is not 1!')

      total_loss= gender_weight * loss_gender + hat_weight * loss_hat + bag_weight * loss_bag
      return total_loss