import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk


def train(dataloader, model, loss_function, optimizer):
    """

    """
    size = len(dataloader.dataset)

    # Set model in training mode, dropout & batch norm layers will be in effect
    model.train(mode=True)

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_prediction = model(X).squeeze(1) # ensure the shape is (batch_size) rather than ([1 batch_size])
        loss = loss_function(y_prediction, y)

        # Back propogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Batch:{batch}|loss: {loss:8f} [{current}|{size}]")

def validate(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_of_batches = len(dataloader)

    model.eval() # Set model in evaluation mode, this disables dropout, and batch norm

    test_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_prediction = model(X).squeeze(1)

        test_loss = loss_function(y_prediction, y)

        correct += ((torch.sigmoid(y_prediction) > 0.5).float() == y).sum().item()
    test_loss /= num_of_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




if __name__=='__main__':
    # If a GPU (NVIDIA, AMD, Apple M1/M2) is detected, the model and tensors will be loaded onto it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





