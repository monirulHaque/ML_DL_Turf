This repository contains my notes of Machine Learning and Deep Learning Concepts.
</br>
-------------------------------------------------------------------------
**Table of Contents**
- [Machine Learning](#machine-learning)
  - [Comparisons](#comparisons)
    - [Supervised vs Unsupervised learning](#supervised-vs-unsupervised-learning)
    - [Regression vs Classificaiton](#regression-vs-classificaiton)
    - [Generative vs discriminative Models](#generative-vs-discriminative-models)
    - [Parametric vs Non-parametric Models](#parametric-vs-non-parametric-models)
    - [Batch vs Online Learning](#batch-vs-online-learning)
    - [Instance-based vs Model-based learning](#instance-based-vs-model-based-learning)
  - [Challenges](#challenges)
    - [Data](#data)
        - [Sampling Techniques](#sampling-techniques)
          - [Uniform sampling](#uniform-sampling)
          - [Reservoir sampling](#reservoir-sampling)
          - [Stratified sampling](#stratified-sampling)
          - [Bootstrapping](#bootstrapping)
          - [Bagging](#bagging)
      - [Non-representive data](#non-representive-data)
      - [Sampling Bias](#sampling-bias)
      - [Missing Data](#missing-data)
      - [Outliers](#outliers)
      - [Imbalanced Data](#imbalanced-data)
        - [Resampling Techniques](#resampling-techniques)
        - [Algorithm-Level Techniques](#algorithm-level-techniques)
        - [Hybrid Strategies](#hybrid-strategies)
        - [Practical Considerations](#practical-considerations)
      - [Data distribution shifts](#data-distribution-shifts)
    - [Feature](#feature)
      - [Feature Selection](#feature-selection)
      - [Feature Extraction](#feature-extraction)
    - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
      - [Regularization Techniques](#regularization-techniques)
        - [L1 Regularization (Lasso)](#l1-regularization-lasso)
        - [L2 Regularization (Ridge)](#l2-regularization-ridge)
        - [Dropout](#dropout)
          - [Standard Dropout](#standard-dropout)
          - [DropConnect](#dropconnect)
          - [Spatial Dropout](#spatial-dropout)
          - [AlphaDropout](#alphadropout)
          - [Variational Dropout](#variational-dropout)
          - [Monte Carlo Dropout](#monte-carlo-dropout)
          - [Weak and Strong Dilution](#weak-and-strong-dilution)
        - [Early Stopping](#early-stopping)
        - [Data Augmentation](#data-augmentation)
        - [Batch Normalization](#batch-normalization)
        - [Adding Weight Constraints](#adding-weight-constraints)
        - [Weight Decay](#weight-decay)
    - [Overfitting vs Underfitting](#overfitting-vs-underfitting)
    - [Curse of Dimensionality](#curse-of-dimensionality)
  - [Testing and Validation](#testing-and-validation)
    - [No Free Lunch Theorem](#no-free-lunch-theorem)
      - [Cross Entropy](#cross-entropy)
    - [Evaluation Metrics](#evaluation-metrics)
      - [Accuracy](#accuracy)
      - [Precision](#precision)
      - [Recall](#recall)
      - [F1-score](#f1-score)
      - [Confusion Matrix](#confusion-matrix)
      - [Log Loss](#log-loss)
      - [AUC – ROC (Area Under the ROC Curve)](#auc--roc-area-under-the-roc-curve)
      - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
      - [Root Mean Squared Logarithmic Error](#root-mean-squared-logarithmic-error)
      - [R-Squared/Adjusted R-Squared](#r-squaredadjusted-r-squared)
      - [Gini Coefficient](#gini-coefficient)
      - [Gain and Lift Charts](#gain-and-lift-charts)
      - [Kolomogorov Smirnov Chart](#kolomogorov-smirnov-chart)
      - [Concordant – Discordant Ratio](#concordant--discordant-ratio)
    - [Hyperparameter Turing](#hyperparameter-turing)
    - [Cross Validation](#cross-validation)
  - [Interpretability \& Explainability](#interpretability--explainability)
  - [Ensemble Learning](#ensemble-learning)
    - [Bagging](#bagging-1)
    - [Boosting](#boosting)
      - [Adaboost](#adaboost)
      - [GBM](#gbm)
      - [XGBoost](#xgboost)
    - [Stacking](#stacking)
    - [Voting](#voting)
    - [Blending](#blending)
    - [Weighted-average](#weighted-average)
- [Machine Learning Algorithms](#machine-learning-algorithms)
  - [Supervised Learning](#supervised-learning)
    - [Linear Regression](#linear-regression)
    - [Logistic Regression](#logistic-regression)
  - [Unsupervised Learning](#unsupervised-learning)
    - [Clustering](#clustering)
    - [Dimensionality Reduction](#dimensionality-reduction)
      - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [Autoencoder](#autoencoder)
- [Deep Learning](#deep-learning)
  - [Activation Functions](#activation-functions)
    - [Sigmoid](#sigmoid)
    - [Tanh](#tanh)
    - [ReLU](#relu)
    - [Leaky ReLU](#leaky-relu)
    - [ELU](#elu)
    - [GeLU](#gelu)
    - [PreLU](#prelu)
    - [Softmax](#softmax)
    - [Loss Functions](#loss-functions)
    - [Logistic Loss function](#logistic-loss-function)
    - [Cross Entropy](#cross-entropy-1)
    - [Hinge loss (SVM)](#hinge-loss-svm)
  - [Optimization](#optimization)
    - [Gradient Descent Algorithm](#gradient-descent-algorithm)
      - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
      - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Momentum](#momentum)
    - [RMSprop](#rmsprop)
    - [Adam](#adam)
  - [Neural Network Structures](#neural-network-structures)
    - [MLP](#mlp)
    - [CNN](#cnn)
    - [RNN](#rnn)
    - [LSTM](#lstm)
    - [Seq2Seq](#seq2seq)
    - [Transformers](#transformers)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Generative Adversarial Network](#generative-adversarial-network)
  - [One-shot and Few Shot learning](#one-shot-and-few-shot-learning)
  - [NLP](#nlp)
  - [Computer Vision](#computer-vision)
- [Latest Tech](#latest-tech)

# Machine Learning
Machine Learning is a very sophisticated form of statistical data that involves predictions and decisions based on data. Rather than specifically programming we let machine learning learn from the data.

## Comparisons
### Supervised vs Unsupervised learning

| Aspect | Supervised Learning | Unsupervised Learning |
| :-- | :-- | :-- |
| **Definition** | Learns from labeled data, mapping inputs to known outputs. | Finds patterns or structures in unlabeled data. |
| **Goal** | Predict or classify outputs for new data. | Discover hidden patterns, groupings, or relationships in the data. |
| **Human Involvement** | Significant (for data labeling and model validation). | Minimal; mainly for validation and algorithm selection. |
| **Primary Tasks** | Classification, Regression. | Clustering, Association, Dimensionality Reduction. 
| **Evaluation** | Metrics like accuracy, precision, recall, F1-score (classification), MSE (regression). | Metrics like silhouette score (clustering), explained variance (dimensionality reduction). |
| **Overfitting Concerns** | Prone to overfitting if not regularized. | Less prone, but can still overfit to noise or irrelevant patterns. |
| **Applications** | Image classification, sentiment analysis, fraud detection, price prediction, Spam detection, medical diagnosis, stock price prediction. | Customer segmentation, anomaly detection, recommendation engines, market basket analysis, Market segmentation, document clustering, anomaly detection in network traffic. |

### Regression vs Classificaiton

| Aspect | Regression | Classification |
| :-- | :-- | :-- |
| **Output Type** | Continuous numerical values. | Discrete (categorical) classes. |
| **Goal** | Predict a continuous value. | Assign input to a predefined category or class. |
| **Decision Boundary** | Predicts a value along a continuum (best-fit line/curve). | Finds decision boundaries to separate classes (sharp transitions). |
| **Further Division** | Linear and Non-linear Regression. | Binary, Multiclass, and Multilabel Classification. |
| **Example Algorithms** | Linear Regression, Regression Trees, SVR. | Logistic Regression, Decision Trees, SVM, k-NN. |
| **Evaluation Metrics** | Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score, MAE. | Accuracy, Precision, Recall, F1 Score, ROC-AUC. |
| **Examples** | Predicting house prices, stock values, temperature. | Classifying spam emails, fake news, diagnosing diseases, image recognition. |

### Generative vs discriminative Models

| Aspect | Generative Models | Discriminative Models |
| :-- | :-- | :-- |
| **What** | Given a set of example data points, D, and their associated labels, L, a generative model learns the joint probability distribution $P(D, L)$  | Given a training dataset consisting of data points, D, and their associated labels, L, a discriminative model learns the conditional probability distribution $P(D \| L)$ |
| **Goal** | Model the distribution of each class and generate new examples | Categorize or predict the class label for new data |
| **Algorithm Examples** | Naive Bayes, Hidden Markov Models, Gaussian Mixture Models, GANs, LDA | Logistic Regression, SVM, Decision Trees, Random Forests, Neural Networks |
| **Robustness to Outliers** | More sensitive to outliers and model assumptions | More robust to outliers and less sensitive to incorrect distribution assumptions |
| **Main Limitation** | Requires strong assumptions about data distribution; can be complex to train | Cannot generate new data; may not capture underlying data structure |

### Parametric vs Non-parametric Models

| Aspect | Parametric Models | Non-parametric Models |
| :-- | :-- | :-- |
| **Definition** | Summarize data with a fixed, finite set of parameters (independent of training data size) | Number of parameters grows with data; model complexity can increase with more data |
| **Flexibility** | Less flexible; limited by the chosen form and number of parameters | More flexible; can adapt to more complex patterns as data increases |
| **Memory Usage** | Low; only need to store the fixed set of parameters | High; may need to store all or a large portion of the training data |
| **Training Data Impact** | Model size and structure do not change with more data | Model can become more complex and accurate as more data is added |
| **Overfitting Risk** | Can underfit if model is too simple for the data | Can overfit if not properly regularized or pruned |
| **Scalability** | Scales well to large datasets if model is appropriate | May not scale well due to increased complexity and memory usage |
| **Example Models** | Linear Regression, Logistic Regression, Neural Networks (with fixed architecture), Naive Bayes | k-Nearest Neighbors, Decision Trees, Random Forests, Gaussian Processes, Kernel Density Estimation |
| **When to Use** | When data fits assumed distribution or when interpretability and efficiency are priorities | When data is complex, high-dimensional, or distribution is unknown |

### Batch vs Online Learning

### Instance-based vs Model-based learning

---------------------------------------------
## Challenges
### Data
- The idea that data matters more than algorithms for complex problems was popularized by Peter Norvig et al. in a paper titled “The Unreasonable Effectiveness of Data”, published in 2009. However, that small and medium-sized datasets are still very common, and it is not always easy or cheap to get extra training data; so algorithms are still important.
- if training data is full of errors, outliers, and noise (e.g., due to poor
quality measurements), it will make it harder for the system to detect the underlying patterns. It is often well worth the effort to spend time cleaning up training data. 

##### Sampling Techniques
###### Uniform sampling
###### Reservoir sampling
###### Stratified sampling
###### Bootstrapping
###### Bagging
#### Non-representive data
#### Sampling Bias
#### Missing Data
If some instances are missing a few features (e.g., 5% of customers did not specify their age), there are a few solutions,
- **Remove the attribute (column)**:
- **Remove the instance (row)**:
- **Train one model with the feature and one model without it**:

#### Outliers

#### Imbalanced Data
Imbalanced datasets occur when one class (majority) significantly outnumbers another (minority), leading to biased models with high overall accuracy but poor minority-class performance. This is critical in domains like fraud detection (1–2% fraud rate) or medical diagnosis where misclassifying minority cases has severe consequences

##### Resampling Techniques
Adjust class distribution at the data level

- **Oversampling**:
    - **Random Oversampling**: Duplicate minority-class instances. Risk of overfitting.
    - **SMOTE/ADASYN**: Generate synthetic minority samples via interpolation (e.g., k-NN features).
- **Undersampling**:
    - **Random Undersampling**: Remove random majority-class instances. Risk of losing useful information.
    - **Informed Methods**: Use Tomek Links or NearMiss to remove ambiguous or redundant majority samples.
- **Hybrid**: Combine oversampling and undersampling (e.g., SMOTE + Tomek Links).

##### Algorithm-Level Techniques
Modify learning algorithms to prioritize minority classes

- **Class Weighting**: Assign higher misclassification costs to minority classes (e.g., `class_weight='balanced'` in scikit-learn).
- **Cost-Sensitive Learning**: Optimize for custom cost matrices where minority misclassification carries higher penalties.
- **Robust Algorithms**:
    - Tree-based methods (Random Forest, XGBoost) with class weighting.
    - SVM with adjusted class weights or focal loss in neural networks.
    - Anomaly detection for extreme imbalance (e.g., <0.1% minority).

##### Hybrid Strategies

- **Downsampling + Upweighting**: Reduce majority samples and assign higher weights to retained instances (e.g., 10× weight if downsampled 10×).
- **Ensemble Methods**: Boost minority-class performance via bagging (e.g., BalancedRandomForestClassifier) or boosting (e.g., AdaBoost with minority focus).
- **Resampling with Cross-Validation**: Apply resampling *within* folds to avoid data leakage.


##### Practical Considerations

- **Preserve Real-World Ratios**: In cases like spam detection, maintain natural imbalance to reflect operational conditions.
- **Data Quantity**: Use oversampling when data is scarce; undersampling when abundant.
- **Algorithm Tuning**: Combine techniques (e.g., SMOTE + class weighting) for severe imbalance.

> "Balancing datasets prevents bias toward the majority class, improving model accuracy and ensuring fair predictions across all classes."


#### Data distribution shifts

### Feature
#### Feature Selection
- **Filter Methods:** These methods assess the relevance of features based on statistical properties such as correlation, chi-square test, or information gain.
- **Wrapper Methods:** These methods evaluate subsets of features by training models iteratively and selecting the best subset based on model performance.
- **Embedded Methods:** These techniques incorporate feature selection as part of the model training process, such as regularization methods like Lasso (L1) or Ridge (L2) regression.
- **Principal Component Analysis (PCA):** A dimensionality reduction technique that identifies linear combinations of features that capture the most variance in the data.
- **Recursive Feature Elimination (RFE):** An iterative technique that recursively removes features with the least importance until the desired number of features is reached.
- **Tree-based Methods:** These methods, such as Random Forest or Gradient Boosting, provide feature importance scores that can be used for selection.
- **Univariate Feature Selection:** Selects features based on univariate statistical tests applied to each feature individually.
#### Feature Extraction

### Bias-Variance Tradeoff
#### Regularization Techniques
##### L1 Regularization (Lasso)
##### L2 Regularization (Ridge)
##### Dropout
> Prevents overfitting by randomly deactivating a subset of neurons during training

###### Standard Dropout

- Randomly sets a fraction (e.g., 20–50%) of neurons' outputs to zero during each training iteration.
- The dropout rate (e.g., 0.5) is a hyperparameter that controls the proportion of dropped units.

###### DropConnect

- Instead of dropping neuron outputs, DropConnect randomly drops (sets to zero) individual weights (connections) between neurons, effectively diluting the connectivity of the network.
- This creates a different random subnetwork at each iteration, similar to standard dropout but at the weight level.

###### Spatial Dropout

- Used mainly in convolutional neural networks (CNNs).
- Instead of dropping individual activations, entire feature maps (channels) are dropped, which helps preserve spatial structure and is more effective for image data.

###### AlphaDropout

- Designed for use with self-normalizing neural networks (e.g., with SELU activation).
- Maintains the mean and variance of inputs, allowing the network to retain its self-normalizing properties.

###### Variational Dropout

- Applies a learned dropout rate for each neuron or weight, often used in Bayesian neural networks.
- The dropout rate becomes a parameter that can be optimized during training.

###### Monte Carlo Dropout

- Dropout is applied during both training and inference.
- Multiple stochastic forward passes are used at test time to estimate uncertainty (commonly used in Bayesian deep learning).

###### Weak and Strong Dilution

- In the context of DropConnect (also called dilution), weak dilution refers to dropping a small fraction of weights, while strong dilution drops a larger fraction.
  

##### Early Stopping
##### Data Augmentation
##### Batch Normalization
##### Adding Weight Constraints
##### Weight Decay
### Overfitting vs Underfitting
### Curse of Dimensionality
---------------------------------------------
## Testing and Validation
### No Free Lunch Theorem
#### Cross Entropy
### Evaluation Metrics
#### Accuracy
#### Precision
#### Recall
#### F1-score
#### Confusion Matrix
#### Log Loss
#### AUC – ROC (Area Under the ROC Curve)
#### Root Mean Squared Error (RMSE)
#### Root Mean Squared Logarithmic Error
#### R-Squared/Adjusted R-Squared
#### Gini Coefficient
#### Gain and Lift Charts
#### Kolomogorov Smirnov Chart
#### Concordant – Discordant Ratio
### Hyperparameter Turing
### Cross Validation

## Interpretability & Explainability

## Ensemble Learning
### Bagging
### Boosting
#### Adaboost
#### GBM
#### XGBoost
### Stacking
### Voting
### Blending
### Weighted-average


# Machine Learning Algorithms
## Supervised Learning
### Linear Regression
### Logistic Regression
## Unsupervised Learning
### Clustering
### Dimensionality Reduction
#### Principal Component Analysis (PCA)
### Autoencoder

# Deep Learning
Deep Learning is a subset of Machine Learning which involves training artificial neural networks on large amount of data. In order to identify and learn hidden patterns in a non-linear data that traditional machine learning algorithms fail to capture. The heart of deep learning is multiple layers of neural networks.

## Activation Functions
### Sigmoid
### Tanh
### ReLU
### Leaky ReLU
### ELU
### GeLU
### PreLU
### Softmax

### Loss Functions
### Logistic Loss function
### Cross Entropy
### Hinge loss (SVM)

## Optimization
### Gradient Descent Algorithm
#### Mini-Batch Gradient Descent
#### Stochastic Gradient Descent
### Momentum
### RMSprop
### Adam

## Neural Network Structures 
### MLP
### CNN
### RNN
### LSTM
### Seq2Seq
### Transformers

## Reinforcement Learning
## Generative Adversarial Network
## One-shot and Few Shot learning

## NLP
## Computer Vision

# Latest Tech
