# Malaria Detection System Using Deep Learning
A comprehensive deep learning solution for automated malaria parasite detection from cell microscopy images. This project implements and compares multiple CNN architectures, explores model explainability through Grad-CAM, and estimates prediction confidence using Monte Carlo Dropout. An interactive Streamlit dashboard provides real-time predictions with uncertainty quantification.

Research Focus: Model comparison, transfer learning, explainability (XAI), and uncertainty estimation in medical imaging.

**Features**

1. Multiple Model Architectures: Baseline LeNet with Batch Normalization, transfer learning with MobileNetV2, and fine-tuned MobileNetV2
2. Comprehensive Model Comparison: Detailed accuracy and loss analysis across training, validation, and test sets
3. Explainability (Grad-CAM): Visual heatmaps showing which regions of cell images influence model predictions
4. Uncertainty Quantification: Monte Carlo Dropout for estimating prediction confidence and model uncertainty
5. Interactive Dashboard: Streamlit-based web application for model comparison, visualization, and inference
6. Real-Time Predictions: Upload images and receive predictions with confidence scores
7. Binary Classification: Parasitized vs. Uninfected cell classification

**Model Performance Comparison**

<img width="907" height="247" alt="image" src="https://github.com/user-attachments/assets/2cf95a80-889a-4208-84fd-537f3d3b36ac" />
Key Insight: The fine-tuned MobileNetV2 achieves the best test accuracy (95.72%), demonstrating the effectiveness of transfer learning combined with fine-tuning on domain-specific medical data.

**Methodology**

**Data**
Source: TensorFlow Malaria Dataset
Classes: Parasitized (infected), Uninfected (clean)
Image Resolution: 224 × 224 pixels (resized for model input)
Train/Val/Test Split: Stratified division to maintain class balance

**Model Development Pipeline**

1. Baseline Model (LeNet with Batch Normalization)
Custom CNN architecture with batch normalization layers
Lightweight, interpretable baseline for comparison
Serves as a reference for transfer learning benefits

2. Transfer Learning (MobileNetV2)
Pre-trained on ImageNet weights
Frozen initial layers to preserve learned features
Fine-tuned dense layers for binary classification

3. Fine-Tuning (MobileNetV2)
Unfroze selective layers for domain adaptation
Low learning rate to prevent catastrophic forgetting
Data augmentation techniques applied during training

**Training Configuration**

1. Optimizer: Adam
2. Loss Function: Binary Crossentropy
3. Batch Size: 32
4. Data Preprocessing: Normalization using MobileNetV2 preprocess_input

**Explainability: Grad-CAM**
Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to provide interpretable predictions by visualizing the regions of input images that most influence the model's decisions.

Approach:
Computes gradients of the output class with respect to feature maps in the final convolutional layer
Generates activation heatmaps highlighting discriminative regions
Heatmaps are overlaid on original images for intuitive visualization

Use Cases:
Clinical Validation: Medical professionals can verify that the model focuses on relevant morphological features (e.g., parasite structures within RBCs)
Model Debugging: Identifies if the model relies on artifacts or irrelevant regions
Trust & Transparency: Builds confidence in model decisions for medical deployment

Current Status:
Grad-CAM heatmaps have been generated for sample images in the exploratory phase
Future Enhancement: Real-time Grad-CAM integration for live predictions (see Limitations & Future Work)

**Uncertainty Estimation: Monte Carlo Dropout**
Monte Carlo Dropout was implemented to quantify model uncertainty, providing confidence intervals alongside predictions.

Approach:
Runs multiple forward passes with dropout enabled during inference
Each forward pass produces slightly different predictions due to stochastic dropout
Aggregates predictions to compute:

Mean Prediction: Expected class probability
Standard Deviation: Uncertainty measure

Interpretation:
Low Uncertainty: Model is confident in its prediction
High Uncertainty: Model is less confident; prediction should be reviewed with caution
Particularly useful for flagging borderline cases in clinical settings

Mathematical Foundation
y^=1N∑i=1Nfdropout(x)\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_{\text{dropout}}(x)y^​=N1​i=1∑N​fdropout​(x)
σ=1N∑i=1N(fdropout(x)−y^)2\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (f_{\text{dropout}}(x) - \hat{y})^2}σ=N1​i=1∑N​(fdropout​(x)−y^​)2​
Where N = 30 forward passes

