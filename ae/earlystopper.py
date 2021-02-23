import torch

"""
A class that tracks the progress of a network. 
Initialised with patience = amount of validations to wait without loss improvement before stopping, delta = the necessary amount of improvement,
save_path = where the model is saved if save=True
"""
class EarlyStopper(): #patience is amount of validations to wait without loss improvment before stopping, delta is the necessary amount of improvement
    def __init__(self,patience,delta,save_path,save):
        self.patience = patience
        self.patience_counter = 0
        self.delta = delta
        self.save_path = save_path
        self.best_loss = -1
        self.save = save

    """
    Checks the loss and isaves the model if loss improved (more than delta).
    Returns False or True if model should stop training based on previous knowledge/critereon.
    """
    def earlyStopping(self,loss,model):
        if self.best_loss == -1 or loss < self.best_loss - self.delta : #case loss decreases
            self.best_loss = loss
            if self.save:
                torch.save(model,str(self.save_path))
            self.patience_counter = 0
            return False
        else: #case loss remains the same or increases
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True