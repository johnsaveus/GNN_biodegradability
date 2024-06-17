from data.mol_to_graph import MolecularDataset
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

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
        y.append(data.y.numpy())
        fps.append(data.fingerprint.numpy().squeeze)
    
    return fps, y

X_train, y_train = extract_features(dataset_train)
X_valid, y_valid = extract_features(dataset_valid)
X_test, y_test = extract_features(dataset_test)

C = [0.01, 0.1, 1, 10]
#C = [0.01,0.1,1,10]

best_f1 = 0 
for c in C:
    model = LogisticRegression(C=c)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    f1_train = roc_auc_score(y_train, train_preds)
    f1_valid = roc_auc_score(y_valid, valid_preds)
    print(f"ROC AUC train = {f1_train}")
    print(f"ROC AUC valid = {f1_valid}")
    if f1_valid > best_f1:
        best_f1 = f1_valid
        params = {'C' : c}

model_final = LogisticRegression(C=params['C'])
model_final.fit(X_train, y_train)

test_preds = model_final.predict(X_test)
roc_auc_test = roc_auc_score(y_test, test_preds)
print(f"ROC test = {roc_auc_test}")
conf = confusion_matrix(y_test, test_preds)
print(conf)


# ROC = 0.828