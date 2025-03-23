import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from icecream import ic
import matplotlib.pyplot as plt
from dataset_pipelines import process_csv, balance_dataset

def preprocess(dataset_type="IO4", test=False, balance=False)->tuple:
    """preprocess dataset

    Parameters
    ----------
    dataset_type : ["IO4", "INT03", "SO2", "S04", "MM05", "MM03"]
        
    test         : boolean
        
    balance      : boolean
        

    Returns
    -------
    (DataFrame, list)
        [TODO:description]

    """
    #scaler = StandardScaler()
    #scaler = MinMaxScaler()

    unprocessed_path = f'./csv/dataset_B/{dataset_type}.csv'
    processed_path = f'./csv/processed_B/processed_{dataset_type}.csv'
    if not os.path.isfile(processed_path):
        print(f"Processing the dataset {dataset_type}\n----------------------------")
        process_csv(unprocessed_path, processed_path)
    processed_df = pd.read_csv(processed_path)
    if balance and (not test):
        print(f"Balancing the dataset {dataset_type}\n----------------------------")
        processed_df = balance_dataset(processed_df, minority_class=1, majority_class=0, label_column="taken")
        
    X = processed_df.drop(columns=['taken', 'PC'])
    y = processed_df['taken'].values
    #X = scaler.fit_transform(X)
    X = X.to_numpy()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    print(f"Done Processing {dataset_type}")
    return X_tensor, y_tensor

def plot_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")

class IterativeDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.BatchNorm1d(100),  # Normalize activations
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(100, 82),
            nn.BatchNorm1d(82),  # Normalize activations
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(82, 64),
            nn.BatchNorm1d(64),  # Normalize activations
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),  # Normalize activations
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        logits = self.linear_relu_stack(X)
        return logits

def train(dataloader, model, loss_func, optimizer, scaler=None):
    size = len(dataloader.dataset)
    model.train(mode=True)

    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if scaler:
            with torch.amp.autocast('cuda'):
                y_prediction = model(X).squeeze(1)
                loss = loss_func(y_prediction, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            y_prediction = model(X).squeeze(1)
            loss = loss_func(y_prediction, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        correct += ((torch.sigmoid(y_prediction) > 0.5).float() == y).sum().item()
        train_loss += loss.item()*X.size(0)



        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Batch:{batch}|loss: {loss:8f} [{current}|{size}]")
    accuracy = correct/size
    return accuracy, train_loss

def validate(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_prediction = model(X).squeeze(1)
            test_loss += loss_func(y_prediction, y).item()
            correct += ((torch.sigmoid(y_prediction) > 0.5).float() == y).sum().item()

    total_loss = test_loss/num_batches
    accuracy = correct/size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, total_loss


def test(dataloader, model, train_accuracy=None, valid_accuracy=None, epochs=10): # changed from test(X,y,...)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze(1)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)
            all_preds.extend(y_pred.cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    #Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds)

    # training and validation Accuracy/loss curves
    if (train_accuracy is not None) and (valid_accuracy is not None):
        plt.figure(figsize=(8, 6))

        plt.plot(range(epochs), train_accuracy)
        plt.plot(range(epochs), valid_accuracy)
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")


if __name__=='__main__':
    # If a GPU (NVIDIA, AMD, Apple M1/M2) is detected, the model and tensors will be loaded onto it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda') # autoscale datatype to speed things up
        pin_memory = True
    else:
        scaler = None
        pin_memory = False

    batch_size = 512
    learning_rate = 1e-4


    X_train1, y_train1 = preprocess("I04", test=False, balance=True)
    X_train2, y_train2 = preprocess("S02", test=False, balance=True)
    X_train3, y_train3 = preprocess("MM05", test=False, balance=True)
    X_train4, y_train4 = preprocess("MM03", test=False, balance=True)

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4))

    X_validate, y_validate = preprocess("INT03", test=False, balance=False)

    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4))
    X_test, y_test = preprocess("S04", test=True, balance=False)

    #|-----------------------------------------|
    #|                 Train                   |
    #|-----------------------------------------|
    ic(X_train)
    ic(y_train)
    train_dataset = IterativeDataset(X_train, y_train)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=3)

    #|-----------------------------------------|
    #|                Validate                 |
    #|-----------------------------------------|
    validate_dataset = IterativeDataset(X_validate, y_validate)

    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=3)


    #|-----------------------------------------|
    #|                 Test                    |
    #|-----------------------------------------|
    test_dataset = IterativeDataset(X_test, y_test)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=3)

    model = NeuralNetwork(X_train.shape[1]).to(device)
    try:
        model.load_state_dict(torch.load('branch_predictor.pth', weights_only=True))
        print("loaded state dict")
    except Exception as e:
        print("Cannot find state dict to load")

    ic(model)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-3) # L2 regularization 
    
    # Do initial training
    init_train_accuracy = []
    init_valid_accuracy = []
    epochs = 25
    for i in range(epochs):
        print(f"Epoch {i+1}\n--------------------------")
        train_accuracy, _ = train(train_dataloader, model, loss_func, optimizer, scaler)
        valid_accuracy, _ = validate(validate_dataloader, model, loss_func)
        init_train_accuracy.append(train_accuracy)
        init_valid_accuracy.append(valid_accuracy)


    print("TESTING\n---------------------------")
    test(test_dataloader, model, init_train_accuracy, init_valid_accuracy, epochs)
    plt.show()
