# Retrosynthesis Prediction with Transformer

This project implements a Transformer-based model for predicting reactant molecules (retrosynthesis) given a target molecule represented as a SMILES string. The model is built using TensorFlow and leverages sequence-to-sequence learning to map product SMILES to reactant SMILES.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Reactants](#predicting-reactants)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [TensorFlow Learning Path for Beginners](#tensorflow-learning-path-for-beginners)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Other Python packages as listed in `requirements.txt`

Before diving into this project, you should have a good understanding of basic neural networks and how they are implemented in TensorFlow. If you are a beginner or want to refresh your knowledge, follow the tutorials listed in the [TensorFlow Learning Path for Beginners](#tensorflow-learning-path-for-beginners).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/deepbiolab/retrosynthesis-transformer.git
   cd retrosynthesis-transformer
   ```

2. **Create a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script can be run in two modes: `train` and `predict`. Use the command-line arguments to specify the desired mode and other parameters.

### Training the Model

To train the Transformer model on your dataset:

```bash
python main.py --mode train \
               --train_file data/retrosynthesis-train.smi \
               --valid_file data/retrosynthesis-valid.smi \
               --epochs 20 \
               --batch_size 64 \
               --checkpoint_dir ./checkpoints
```

**Arguments:**

- `--mode`: Set to `train` to initiate training.
- `--train_file`: Path to the training data file in SMILES format.
- `--valid_file`: Path to the validation data file in SMILES format.
- `--epochs`: Number of epochs to train the model (default: 20).
- `--batch_size`: Batch size for training (default: 64).
- `--checkpoint_dir`: Directory to save model checkpoints (default: `./checkpoints`).

**Example:**

```bash
python main.py --mode train --epochs 30
```

This command trains the model for 30 epochs using the default training and validation files, batch size, and checkpoint directory.

### Predicting Reactants

After training, you can use the model to predict reactants for a given product SMILES string.

```bash
python main.py --mode predict \
               --input_sequence "Ic1ccc2n(CC(=O)N3CCCCC3)c3CCN(C)Cc3c2c1" \
               --checkpoint_dir ./checkpoints
```

**Arguments:**

- `--mode`: Set to `predict` to perform prediction.
- `--input_sequence`: The product SMILES string for which to predict reactants.
- `--checkpoint_dir`: Directory where model checkpoints are saved (default: `./checkpoints`).

**Example:**

```bash
python main.py --mode predict --input_sequence "Cc1ccccc1O"
```

**Output:**

```
Input Product:       Cc1ccccc1O
Predicted Reactants: Cc1ccccc1 + [Another Reactant]
```

**Note:** Ensure that the model has been trained and the checkpoints are available before performing predictions.

## Data Format

- **Training and Validation Files:** The data files should contain SMILES strings, each representing a chemical reaction in the format:

  ```
  Reactant1.Reactant2>>Product
  ```

  Each line corresponds to a single reaction.

## Project Structure

```
retrosynthesis-transformer/
│
├── data/
│   ├── retrosynthesis-train.smi
│   └── retrosynthesis-valid.smi
│
├── checkpoints/
│   └── ... (model checkpoints)
│
├── transformer.py
├── utils.py
├── preprocess.py
├── main.py
└── requirements.txt
```

- `data/`: Contains training and validation data files.
- `checkpoints/`: Stores model checkpoints during training.
- `transformer.py`: Defines the Transformer model architecture.
- `utils.py`: Contains utility functions for masks, loss calculation, checkpoint management, etc.
- `preprocess.py`: Handles data preprocessing, tokenization, and dataset preparation.
- `main.py`: The main script to train and predict using the Transformer model.
- `requirements.txt`: Lists all Python dependencies.
- `README.md`: Project documentation.

## TensorFlow Learning Path for Beginners

To help beginners understand the fundamentals of neural networks and how to use TensorFlow, we've provided a learning path. You are encouraged to complete these tutorials before diving into this project:

### Tier 1: Neural Network Basics

1. **[00-Neural Network Basic](./neural_network_basics/00-Neural%20Network%20Basic.ipynb)**

   - From Neuron to Single Layer Perceptrons
   - From Single Layer Perceptrons to Multi-Layer Perceptron 
   - From Multi-Layer Perceptron to Deep Neural Network 

2. **[01-Keras Basic](./neural_network_basics/01-Keras%20Basic.ipynb)**

   - Introduction to Keras
   - Model creation methods
   - Viewing model details
   - Defining loss functions, optimizers, and compiling the model
   - Training, evaluating, and predicting with the model
   - Saving and restoring models
   - Callbacks
   - Visualizing training with TensorBoard
   - Parameter tuning

3. **[02-TensorFlow Basic](./neural_network_basics/02-TensorFlow%20Basic.ipynb)**

   - Introduction to TensorFlow
   - Tensor operations

4. **[03-Training Techniques](./neural_network_basics/03-Training%20Techniques.ipynb)**

   - Initialization methods
   - Non-saturating activation functions
   - Batch normalization
   - Gradient clipping
   - Optimizer selection
   - Learning rate scheduling
   - Regularization techniques

### Tier 2: Transformer Implementation

Once you understand the basics of neural networks, follow this project's code to learn how to implement a Transformer model, which is used for advanced sequence-to-sequence tasks such as retrosynthesis prediction.


### Additional Notes

- **Data Preparation:** Ensure that your training and validation data are properly formatted and located in the specified paths. The script assumes that the data files are in SMILES format suitable for retrosynthesis tasks.

- **Hardware Requirements:** Training Transformer models can be computationally intensive. It's recommended to use a machine with a GPU to accelerate the training process.

- **Extensibility:** The model and training pipeline can be further enhanced by incorporating more sophisticated tokenization, data augmentation, or by experimenting with different hyperparameters.

---
