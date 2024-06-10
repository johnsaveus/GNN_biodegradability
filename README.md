# Molecular Graph Classification
 
This repo contains the semester MSc project for the course of Deep Learning. It's about using Graph Neural Network to classify molecules into readily biodegradable and non-readily biodegredable. The Graph Neural Network will contain attention message-passing and will be compared to machine learning methods. ?? Also an architecture FP-GNN will be implemented ??

### Label information

Ready-biodegradability: The ability of a substance to biodegrade quickly >60% within a 28-day window. Readily biodegradable products are environmentally friendly and preferable.

### Datasets

Merge of two datasets containing a total of 6537 compounds:
1) https://zenodo.org/records/3540701
2) https://zenodo.org/records/8255910

### Modelling and scripts

- data_splitter.py: Stratified Split into train-valdiation-test with 80,10,10 ratio. Dataset is imbalanced 55-45 and all the sets contain this ratio.

- mol_to_graph.py: Creates the datasets that contain the node features-signal, adjacency matrix, label, ...Node signal per node = 68, containing information about atoms, neighbors, formal_charge  etc... 

- gat_layer.py: Contains the message passing with attention.

- gat_network.py: Contains the Graph Neural Network architecture with Message Passing layers.

- gat_tuning.py: Training and evaluation loop (validation set).
 
