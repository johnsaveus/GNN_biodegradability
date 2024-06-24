from data.mol_to_graph import MolecularDataset
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import seaborn as sns

dataset_train = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_train_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
                                save_name='train')
dataset_valid = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_valid_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_valid_ix.txt',
                                save_name='valid')
dataset_test = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_test_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_test_ix.txt',
                                save_name='test')
dataset_train.load("data/processed/train.pt")
dataset_valid.load("data/processed/valid.pt")
dataset_test.load("data/processed/test.pt")

def extract_features(dataset):
    y = []
    fps = []
    for data in dataset:
        fp = data.fingerprint.numpy().squeeze()
        y.append(data.y.numpy())
        fps.append(fp)
    
    return fps, y

X_train, y_train = extract_features(dataset_train)
X_valid, y_valid = extract_features(dataset_valid)
X_test, y_test = extract_features(dataset_test)

C_logistic = [0.01, 0.1, 1, 10]

best_ba = 0 
for c in C_logistic:
    model = LogisticRegression(C=c)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    ba_train = balanced_accuracy_score(y_train, train_preds)
    ba_valid = balanced_accuracy_score(y_valid, valid_preds)
    print(f"BA train = {ba_train}")
    print(f"BA valid = {ba_valid}")
    if ba_valid > best_ba:
        best_ba = ba_valid
        params = {'C' : c}

model_final = LogisticRegression(C=params['C'])
model_final.fit(X_train, y_train)
test_preds = model_final.predict(X_test)
ba = balanced_accuracy_score(y_test, test_preds)
conf = confusion_matrix(y_test, test_preds)

class_names = ['NRB', 'RB']
logistic_path = 'results/Logistic'

if not os.path.exists(logistic_path):
    os.makedirs(logistic_path)

file_path = os.path.join(logistic_path, 'best_params.txt')

# Write the best parameters to the file
with open(file_path, 'w') as file:
    file.write("Best Logistic Regression Parameters:\n")
    for param, value in params.items():
        file.write(f"{param}: {value}\n")
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Logistic Regression Balanced Accuracy: {ba:.3f}')
plt.savefig(logistic_path + '/test_cf.png')

C_svm = [0.01, 0.1, 1, 10]
kernel = ['linear', 'rbf', 'poly']
best_ba = 0 
for c in C_svm:
    for k in kernel:
        model = SVC(C=c, kernel=k)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        valid_preds = model.predict(X_valid)
        ba_train = balanced_accuracy_score(y_train, train_preds)
        ba_valid = balanced_accuracy_score(y_valid, valid_preds)
        print(f"BA train = {ba_train}")
        print(f"BA valid = {ba_valid}")
        if ba_valid > best_ba:
            best_ba = ba_valid
            params = {'C' : c,
                      'kernel' : k}
            
model_final = SVC(C=params['C'], kernel= params['kernel'])
model_final.fit(X_train, y_train)
test_preds = model_final.predict(X_test)
ba = balanced_accuracy_score(y_test, test_preds)
conf = confusion_matrix(y_test, test_preds)

svm_path = 'results/SVM'

if not os.path.exists(svm_path):
    os.makedirs(svm_path)

file_path = os.path.join(svm_path, 'best_params.txt')

# Write the best parameters to the file
with open(file_path, 'w') as file:
    file.write("Best Logistic Regression Parameters:\n")
    for param, value in params.items():
        file.write(f"{param}: {value}\n")
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'SVM Balanced Accuracy: {ba:.3f}')
plt.savefig(svm_path + '/test_cf.png')