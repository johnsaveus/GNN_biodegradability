from data.mol_to_graph import MolecularDataset
from Net.fp_gnn import FP_GNN
import json
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataset = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_test_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_test_ix.txt',
                                save_name='test')
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = 1)
model_path = 'results/Best_fpnn'
with open(model_path + '/hyperparams.json', 'r') as file:
    params = json.load(file)

model = model = FP_GNN(hidden_feats=params['hidden'],
                       num_heads=params['num_heads'],
                       num_layers=params['num_layers'],
                       activation = params['activation'],
                       drop_prob =params['dropout']).to(device)

model.load_state_dict(torch.load(model_path + '/Model.pth'))

model.eval()
preds = []
targets = []
with torch.no_grad():
    for data in test_dataset:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        fp = data.fingerprint.to(device)
        batch = data.batch
        y_logit = model(x, edge_index, fp, batch)
        y_sigm = torch.sigmoid(y_logit)
        preds.append((y_sigm > 0.5).int().cpu())
        targets.append(data.y.int())
print(balanced_accuracy_score(targets, preds))
class_names = ['NRB', 'RB']
conf = confusion_matrix(targets, preds)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(model_path + '/test_cf.png')
plt.show()