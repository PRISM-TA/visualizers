import torch
import torch.nn as nn


class BaseHyperParam:

    num_epochs: int # number of epochs for which the model will be trained

    early_stopping: bool # boolean flag indicating whether early stopping should be applied
    val_loader: torch.utils.data.DataLoader # data loader for the validation dataset, required if early stopping is enabled
    patience: int # number of epochs with no improvement after which training will be stopped if early stopping is on

    def __init__(self, 
                 num_epochs: int = 100, 
                 early_stopping: bool = True, 
                 patience: int = 50):
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
    
    def __repr__(self):
        return f"<BaseHyperParam(num_epochs={self.num_epochs}, early_stopping={self.early_stopping}, val_loader={self.val_loader}, patience={self.patience})>"