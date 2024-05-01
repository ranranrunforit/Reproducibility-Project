# Reproducibility Project for Graph Attention Networks

This repo is a reproduction study based on the original publication and repo of the **Graph Attention Networks (GAT)**.  

The publication: *P. VelickoviÊ, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.*.  

Link to the publication: [arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)

Link to the github repository: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

The paper introduces Graph Attention Networks (GAT) as an innovative approach to processing graph-structured data using convolution-style neural networks with masked self-attentional layers. GAT enables nodes to assign varying weights to their neighbors without costly matrix operations or prior knowledge of the entire graph structure. By leveraging attention mechanisms efficiently within the graph attentional layer, GAT achieves computational efficiency and can parallelize operations across all nodes in the graph. This dynamic assignment of node importance based on relationships leads to improved information propagation and feature learning, resulting in state-of-the-art performance on node classification benchmarks. GAT offers a more effective and scalable solution for handling complex relationships in graph data.

This repository contains two PyTorch implementations of GAT model mentioned in the paper: one for transductive learning and one for inductive learning.


# Requirements
- Python 3.10.12
- PyTorch  2.2.1+cu121
- PyTorch Geometric  2.5.3
- NumPy  1.25.2
- scikit-learn  1.2.2
- Ipython  7.34.0
- matplotlib  3.7.1

# Dataset

The project utilized three classic graph datasets, namely Cora, CiteSeer, and Protein-Protein Interactions (PPI) datasets.

For transductive learning, both Cora and CiteSeer Datasets were used, representing networks of research papers with each connection representing a citation. The Cora dataset consists of 2708 scientific publications classified into seven classes, with a citation network of 5429 links. Each publication is described by a word vector (node features) indicating the presence or absence of 1433 unique words in a paper. The CiteSeer dataset consists of 3327 scientific publications classified into six classes, with a citation network of 4732 links. Each publication is described by a word vector indicating the presence or absence of 3703 unique words.

For inductive learning, the PPI dataset contains 24 graphs corresponding to different human tissues, with 20 graphs for training, 2 for validation, and 2 for testing. The average number of nodes per graph is 2372, and each node has 50 features composed of positional gene sets, motif gene sets, and immunological signatures, as well as 121 labels from the Molecular Signatures Database.

# Model Architecture

The model architecture for Graph Attention Networks (GAT) as proposed by Veličković et al. consists of two main model configurations, transductive learning model on Cora and Citeseer datasets, and inductive learning model on the PPI dataset. Both models are initialized using Glorot initialization and trained to minimize cross-entropy on the training nodes using the Adam SGD optimizer with an initial learning rate of 0.005 for all datasets. In both cases, an early stopping strategy is employed on both the cross-entropy loss and accuracy (transductive) or micro-F1 (inductive) score on the validation nodes, with a patience of 100 epochs.

For transductive learning, a two-layer GAT model is applied. The first layer consists of K = 8 attention heads computing F = 8 features each, followed by an exponential linear unit (ELU) nonlinearity. This results in a total of 64 features. The second layer is used for classification and consists of a single attention head that computes C features, where C is the number of classes, followed by a softmax activation. To address the small training set sizes, regularization is applied within the model. Specifically, L2 regularization with lambda = 0.0005 is applied during training. Additionally, dropout with p = 0.6 is applied to both layers’ inputs, as well as to the normalized attention coefficients.
The architectural hyperparameters have been optimized on the Cora dataset and then reused for the Citeseer dataset.

For inductive learning, a three-layer GAT model is applied. Both of the first two layers consist of K = 4 attention heads computing F = 256 features each, followed by an ELU nonlinearity, resulting in a total of 1024 features. The final layer is used for (multi-label) classification and consists of K = 6 attention heads computing 121 features each, which are averaged and followed by a logistic sigmoid activation. In this case, the training sets are sufficiently large, and there is no need to apply L2 regularization or dropout. However, skip connections across the intermediate attentional layer have been successfully employed. A batch size of 2 graphs is utilized during training for this task.


# Device

To significantly speed up training, it's recommended to utilize a GPU. 

If using Google Colab, please click `Runtime` and then `Change runtime type`. Then set the `hardware accelerator` to **GPU**.


# Usage

**It's recommended to run the codes for one model at a time to accurately evaluate each model's performance within the 8-minute time limit.**

**Note**: Make sure to **sequentially run all the cells in each section** so that the intermediate variables / packages will carry over to the next cell

**Install Pytorch Geometric**

PyTorch Geometric library is used to handle graph data.
https://pytorch-geometric.readthedocs.io/en/latest/

- To install PyTorch Geometric in Google Colab, simply run the following cell.
- For other environments, follow the documentation to install Pytorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html and skip the following cell.


Training and evaluating the GAT model on the Cora dataset can be done through running the the `main.py` script as follows:


1. Clone the repository:

```
git clone https://github.com/ebrahimpichka/GAT-pt.git
cd GAT-pt/
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Train the GAT model by running the the `train.py` script as follows:: (Example using the default parameters)

```bash
python train.py --epochs 300 --lr 0.005 --l2 5e-4 --dropout-p 0.6 --num-heads 8 --hidden-dim 64 --val-every 20
```

In more detail, the `main.py` script recieves following arguments:
```
usage: train.py [-h] [--epochs EPOCHS] [--lr LR] [--l2 L2] [--dropout-p DROPOUT_P] [--hidden-dim HIDDEN_DIM] [--num-heads NUM_HEADS] [--concat-heads] [--val-every VAL_EVERY]
               [--no-cuda] [--no-mps] [--dry-run] [--seed S]

PyTorch Graph Attention Network

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 300)
  --lr LR               learning rate (default: 0.005)
  --l2 L2               weight decay (default: 6e-4)
  --dropout-p DROPOUT_P
                        dropout probability (default: 0.6)
  --hidden-dim HIDDEN_DIM
                        dimension of the hidden representation (default: 64)
  --num-heads NUM_HEADS
                        number of the attention heads (default: 4)
  --concat-heads        wether to concatinate attention heads, or average over them (default: False)
  --val-every VAL_EVERY
                        epochs to wait for print training and validation evaluation (default: 20)
  --no-cuda             disables CUDA training
  --no-mps              disables macOS GPU training
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 13)
```

More detailed example of training and evaluating model can be found in `DL4H_Team_33_cz78.ipynb` script .


# Results
In transductive learning, the GAT model was trained for 200 epochs over 10 runs using default hyperparameters on randomly split train/val/test data. The model achieved approximately 82.60% classification accuracy on the test split of the Cora Dataset and 71.53% classification accuracy on the test split of the Citeseer Dataset. These results are comparable to the performance reported in the original paper. The variability in results can be attributed to the randomness of the train/val/test split.

In the case of inductive learning, after 10 runs of training for 200 epochs, the GAT model achieved an average of 97.51% +/- 0.16% classification micro F1 score on the test split.

# Reference

``` 
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
```


![arXiv](./image.png)
