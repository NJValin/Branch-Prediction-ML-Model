import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    columns = [f'Predicted {i}' for i in range(len(cm[0]))]
    index = [f'Actual {i}' for i in range(len(cm[0]))]
    
    cm_df = pd.DataFrame(cm, index=index, columns=columns)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()




    

train_df = pd.read_csv('./csv/processed_B/processed_I04.csv')
a = (train_df.taken.values == 0).sum()
b = (train_df.taken.values == 1).sum()
print("Number of branches taken in I04:", (train_df.taken.values == 1).sum())
print("Number of branches not taken in I04:", (train_df.taken.values == 0).sum())
print("Ratio:", b/a)
print
test_df = pd.read_csv('./csv/processed_B/processed_INT03.csv')

X_train = train_df.drop(columns=['taken'])
X_test = test_df.drop(columns=['taken'])

y_train = train_df['taken'].values
y_test = test_df['taken'].values

classifier = RandomForestClassifier(n_jobs=-1)

model = classifier.fit(X_train, y_train)

y_predicted = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predicted))
print("\nClassification Report:\n", classification_report(y_test, y_predicted))
plot_confusion_matrix(y_test, y_predicted)
