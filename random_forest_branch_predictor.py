import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np
from dataset_pipelines import process_csv, balance_dataset
import os

def preprocess(dataset_type="IO4", test=False, balance=False)->(pd.DataFrame, list):
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

    unprocessed_path = f'./csv/dataset_B/{dataset_type}.csv'
    processed_path = f'./csv/processed_B/processed_{dataset_type}.csv'
    if not os.path.isfile(processed_path):
        print(f"Processing the dataset {dataset_type}\n----------------------------")
        process_csv(unprocessed_path, processed_path)
    processed_df = pd.read_csv(processed_path)
    if balance and (not test):
        print(f"Balancing the dataset {dataset_type}\n----------------------------")
        processed_df = balance_dataset(processed_df, minority_class=1, majority_class=0, label_column="taken")
        
    X = processed_df.drop(columns=['taken'])
    y = processed_df['taken'].values
    #X.loc[:, X.columns.str.startswith('GA_TABLE_')] = X.loc[:, X.columns.str.startswith('GA_TABLE_')].astype(float) / 255 # normalize each entry of each 8-bit GA_Table 
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

def plot_feature_importances(X, y):
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 5))
    plt.barh(features, importances, color="green")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance (Random Forest)")

if __name__=='__main__':
    X_train1, y_train1 = preprocess("I04", test=False, balance=True)
    X_train2, y_train2 = preprocess("S02", test=False, balance=True)
    X_train3, y_train3 = preprocess("MM05", test=False, balance=True)

    X_train = pd.concat([X_train1, X_train2, X_train3]).reset_index(drop=True)
    y_train = np.concatenate((y_train1, y_train2, y_train3))
    

    X_test, y_test = preprocess("S04", test=True, balance=False)

    classifier = RandomForestClassifier(n_estimators=200,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='sqrt',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=-1,
                                 random_state=None,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None,
                                 ccp_alpha=0.0,
                                 max_samples=None)

    print("Fitting!\n----------------------------")
    model = classifier.fit(X_train, y_train)

    print("Predicting!\n----------------------------")
    y_predicted = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_predicted))
    print("\nClassification Report:\n", classification_report(y_test, y_predicted))
    plot_feature_importances(X_train, y_train)
    plot_confusion_matrix(y_test, y_predicted)

    plt.show()
