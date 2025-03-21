# Customer_Churn_Prediction
This Notebook is for predicting if a customer stays with his/her subscription


# Churn Prediction Model

## Overview
This repository contains a deep learning model designed to predict customer churn using a neural network architecture. The model utilizes TensorFlow and Keras with distributed training capabilities to efficiently process and analyze customer data to identify potential churn risks.

## Key Features
- **Multi-GPU Training**: Implemented using TensorFlow's distribution strategy for accelerated training
- **Advanced Neural Network Architecture**: Deep neural network with multiple hidden layers
- **LeakyReLU Activation**: Uses LeakyReLU instead of standard ReLU to prevent "dying ReLU" problem
- **Dropout Regularization**: Applied between layers to prevent overfitting
- **AdamW Optimizer**: Implements weight decay regularization for better generalization
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## Dataset
The model uses the "Churn_Modelling.csv" dataset with the following preprocessing steps:
- Removal of irrelevant columns (RowNumber, CustomerId, Surname)
- One-hot encoding for categorical geographic data
- Label encoding for gender data
- Feature standardization using StandardScaler

## Model Architecture
```
Layer (type)                    Output Shape         Param #
===========================================================
Dense (128 neurons)             (None, 128)          Input layer
LeakyReLU (alpha=0.01)          (None, 128)          Advanced activation
Dropout (0.2)                   (None, 128)          Regularization

Dense (64 neurons)              (None, 64)           Hidden layer 1
LeakyReLU (alpha=0.01)          (None, 64)           Advanced activation
Dropout (0.2)                   (None, 64)           Regularization

Dense (32 neurons)              (None, 32)           Hidden layer 2
LeakyReLU (alpha=0.01)          (None, 32)           Advanced activation
Dropout (0.2)                   (None, 32)           Regularization

Dense (16 neurons)              (None, 16)           Hidden layer 3
LeakyReLU (alpha=0.01)          (None, 16)           Advanced activation
Dropout (0.2)                   (None, 16)           Regularization

Dense (1 neuron, sigmoid)       (None, 1)            Output layer
===========================================================
```

## Training Process
- **Distribution Strategy**: Training leverages multiple GPUs for parallel processing
- **Early Stopping**: Training halts when validation loss stops improving after 10 epochs
- **Batch Size**: 32 samples per batch
- **Validation Split**: 20% of training data used for validation
- **Maximum Epochs**: 100 (with early stopping)

## Evaluation
Model performance is evaluated on a 20% test set using:
- Binary cross-entropy loss
- Accuracy metric

## Usage
1. Load the dataset from /kaggle/input/churnprediction/Churn_Modelling.csv
2. Run the preprocessing steps as outlined in the code
3. Train the model using the fit method
4. Evaluate the model on test data

## Unique Aspects of This Implementation
- **LeakyReLU Activation Function**: Addresses the "dying neuron" problem common with standard ReLU
- **Progressive Layer Reduction**: Neural architecture progressively reduces from 128→64→32→16→1 neurons
- **Multi-GPU Support**: Designed to scale across multiple GPUs for faster training
- **AdamW Optimizer**: Uses modern optimizer with weight decay instead of standard Adam
- **Comprehensive Dropout**: Every layer includes dropout for thorough regularization
- **Strategic Data Encoding**: Different encoding strategies used based on feature characteristics

## Dependencies
- TensorFlow 2.x
- Keras
- Pandas
- Scikit-learn
- NumPy

## Future Improvements
- Hyperparameter tuning via grid search or Bayesian optimization
- Experiment with different network architectures
- Implement additional feature engineering
- Add class weighting for imbalanced data
- Explore ensemble methods combining neural networks with traditional ML algorithms
