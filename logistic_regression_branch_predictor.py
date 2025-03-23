import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np
from dataset_pipelines import process_csv, balance_dataset
import os

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
    scaler = StandardScaler()
    pca = PCA(n_components=32)
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
    X = scaler.fit_transform(X)
    X = pca.fit_transform(X)
    print(f"Done Processing {dataset_type}")
    return X, y


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    columns = [f'Predicted {i}' for i in range(len(cm[0]))]
    index = [f'Actual {i}' for i in range(len(cm[0]))]
    
    cm_df = pd.DataFrame(cm, index=index, columns=columns)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")


if __name__=='__main__':
    X_train1, y_train1 = preprocess("I04", test=False, balance=False)
    X_train2, y_train2 = preprocess("S02", test=False, balance=False)
    X_train3, y_train3 = preprocess("MM05", test=False, balance=False)
    X_train4, y_train4 = preprocess("MM03", test=False, balance=False)
    X_train5, y_train5 = preprocess("INT03", test=False, balance=False)

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5))
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
    

    X_test, y_test = preprocess("S04", test=True, balance=False)

    classifier = LogisticRegression(penalty="l1", solver="saga", C=0.1, class_weight="balanced", n_jobs=-1)

    print("Fitting!\n----------------------------")
    model = classifier.fit(X_train, y_train)

    print("Predicting!\n----------------------------")
    y_predicted = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_predicted))
    print("\nClassification Report:\n", classification_report(y_test, y_predicted))
    plot_confusion_matrix(y_test, y_predicted)

    plt.show()
