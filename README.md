# Molecular Graph Classification
 
This repo contains the semester MSc project for the course of Deep Learning. It's about using Graph Neural Network to classify molecules into readily biodegradable and non-readily biodegredable. The Graph Neural Network will contain attention message-passing and will be compared to machine learning methods. Also FP-GNN was implemented to compare with previous models.

### Label information

Ready-biodegradability: The ability of a substance to biodegrade quickly >60% within a 28-day window. Readily biodegradable products are environmentally friendly and preferable.

### Datasets

Merge of two datasets containing a total of 6537 compounds:
1) https://zenodo.org/records/3540701
2) https://zenodo.org/records/8255910

### Modelling and scripts

- data_splitter.py: Stratified Split into train-valdiation-test with 80,10,10 ratio.

- mol_to_graph.py: Creates the datasets that contain the node features-signal, adjacency matrix, label, and fingerprints.

- ML_baselines.py : ML baselines for comparisons with GNNs

- gat_layer.py: Contains the message passing with attention.

- gat_network.py: Contains the Graph Neural Network architecture with Message Passing layers.

- fp_gnn.py: Contains the FP-GNN architecture.

- gat_tuning.py: Training and evaluation loop (validation set).

- gat_inference.py: Test set innference of the trained model

- fpnn_tuning.py: Training and evaluation loop (validation set).

- fpnn_inference.py: Test set inference of the trained model
 
