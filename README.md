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

**Data**\
Source: TensorFlow Malaria Dataset\
Classes: Parasitized (infected), Uninfected (clean)\
Image Resolution: 224 × 224 pixels (resized for model input)\
Train/Val/Test Split: Stratified division to maintain class balance\

**Model Development Pipeline**

1. Baseline Model (LeNet with Batch Normalization)\
Custom CNN architecture with batch normalization layers\
Lightweight, interpretable baseline for comparison\
Serves as a reference for transfer learning benefits\

2. Transfer Learning (MobileNetV2)\
Pre-trained on ImageNet weights\
Frozen initial layers to preserve learned features\
Fine-tuned dense layers for binary classification\

3. Fine-Tuning (MobileNetV2)\
Unfroze selective layers for domain adaptation\
Low learning rate to prevent catastrophic forgetting\
Data augmentation techniques applied during training\

**Training Configuration**\

1. Optimizer: Adam
2. Loss Function: Binary Crossentropy
3. Batch Size: 32
4. Data Preprocessing: Normalization using MobileNetV2 preprocess_input

**Explainability: Grad-CAM**
Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to provide interpretable predictions by visualizing the regions of input images that most influence the model's decisions.

Approach:\
Computes gradients of the output class with respect to feature maps in the final convolutional layer\
Generates activation heatmaps highlighting discriminative regions\
Heatmaps are overlaid on original images for intuitive visualization\

Use Cases:\
Clinical Validation: Medical professionals can verify that the model focuses on relevant morphological features (e.g., parasite structures within RBCs)\
Model Debugging: Identifies if the model relies on artifacts or irrelevant regions\
Trust & Transparency: Builds confidence in model decisions for medical deployment\

Current Status:\
Grad-CAM heatmaps have been generated for sample images in the exploratory phase\
Future Enhancement: Real-time Grad-CAM integration for live predictions (see Limitations & Future Work)\

**Uncertainty Estimation: Monte Carlo Dropout**\
Monte Carlo Dropout was implemented to quantify model uncertainty, providing confidence intervals alongside predictions.\

Approach:\
Runs multiple forward passes with dropout enabled during inference\
Each forward pass produces slightly different predictions due to stochastic dropout\
Aggregates predictions to compute:\
Mean Prediction: Expected class probability\
Standard Deviation: Uncertainty measure\

Interpretation:\
Low Uncertainty: Model is confident in its prediction\
High Uncertainty: Model is less confident; prediction should be reviewed with caution\
Particularly useful for flagging borderline cases in clinical settings\

Mathematical Foundation\
<img width="386" height="245" alt="image" src="https://github.com/user-attachments/assets/400ad4db-2a38-487e-a2a8-83e9d23f5f71" />
Where N = 30 forward passes\

**Dashboard Overview**\
An interactive Streamlit application provides a comprehensive interface for model evaluation and inference.\
Dashboard Sections\
1. Model Comparison Table\
Side-by-side accuracy metrics for all three models\
Highlights performance improvements across train/val/test sets\

2. Training Curves Visualization\
Loss curves (convergence analysis) for each model\
Accuracy curves (generalization performance) for each model\
Visual comparison of overfitting/underfitting behavior\

3. Explainability Gallery\
Sample Grad-CAM heatmaps for Parasitized cells\
Sample Grad-CAM heatmaps for Uninfected cells\
Original image + heatmap overlay for interpretability\

4. Monte Carlo Uncertainty Estimation\
Sample prediction distributions from Monte Carlo forward passes\
Histogram of prediction probabilities showing model confidence\
Demonstrates uncertainty quantification in action\

5. Live Prediction Interface\
Upload custom cell microscopy images (JPG/PNG)\
Real-time prediction: Parasitized or Uninfected\
Confidence score derived from Monte Carlo mean prediction\
Note: Grad-CAM is not yet applied to live predictions (see Future Work)\

7. Running the Dashboard
streamlit run app.py

**Tech Stack**
<img width="794" height="451" alt="image" src="https://github.com/user-attachments/assets/cf1f4659-f858-4c7e-a631-98e802cb1948" />

**Installation & Setup**

Prerequisites
Python 3.8 or higher
pip or conda package manager
GPU (optional, recommended for training)

Step 1: Clone the Repository
git clone https://github.com/your-username/malaria-detection-system.git
cd malaria-detection-system

Step 2: Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Requirements File
tensorflow==2.13.0
streamlit==1.28.0
numpy==1.24.3
pillow==10.0.0

**Usage**

**Training a Model (Google Colab)**
1. Open the relevant notebook in Google Colab:
malaria_bn.ipynb for LeNet
mobilenetv2.ipynb for MobileNetV2

2. Mount Google Drive and set up the dataset
Run all cells sequentially
Save the trained model to your drive

3. Making Predictions
Using the Dashboard
Run streamlit run app.py
Upload an image in the "Upload Image for Prediction" section
View prediction and confidence score

Key Findings

1. Transfer Learning Effectiveness:
Fine-tuned MobileNetV2 outperformed the custom baseline LeNet model
Test accuracy improved from 93.73% (LeNet) to 95.72% (Fine-tuned MobileNetV2)
Demonstrates the value of pre-trained ImageNet features for medical imaging tasks

2. Generalization
Model maintains strong test performance despite training on limited data
Validation accuracy (96.5%) close to test accuracy (95.72%) indicates good generalization

3. Model Interpretability
Grad-CAM heatmaps reveal that the model focuses on cell interior structures
High confidence in identifying parasitized cells when parasites are present

4. Uncertainty Quantification:
Monte Carlo Dropout captures model uncertainty effectively
Low uncertainty for clear cases; higher uncertainty for ambiguous cell morphologies

**Limitations & Future Work**
**Current Limitations**

1.Grad-CAM Not Integrated in Live Predictions
Heatmaps are shown as pre-generated examples only
Real-time Grad-CAM generation for uploaded images is not yet implemented

2.Limited Production Readiness
Dashboard designed for demonstration; not optimized for clinical deployment
No user authentication, logging, or audit trails

3. Single Dataset
Evaluated only on TensorFlow Malaria Dataset
Generalization to other malaria datasets or imaging modalities untested

4. Binary Classification Only
Does not distinguish between different parasite types or infection stages
Currently limited to Parasitized vs. Uninfected classification

5. Dropout-Based Uncertainty
Monte Carlo Dropout provides uncertainty but is computationally expensive (N=30 forward passes)
Alternative methods (Bayesian approaches) not explored

**Future Work**

1. Real-Time Grad-CAM: Integrate Grad-CAM heatmap generation for live predictions
2. Confidence Thresholding: Implement automatic flagging of low-confidence predictions
3. Multi-Class Classification: Extend to classify different parasite types (Plasmodium species)
4. Model Optimization: Quantization and pruning for edge deployment (mobile/embedded devices)
5. External Validation: Test on independent datasets from different clinical labs
6. Ensemble Methods: Combine multiple models for improved robustness
7. LIME Integration: Implement LIME alongside Grad-CAM for complementary explanations
8. Production API: Develop REST API with proper error handling and logging
9. Performance Benchmarking: Compare with state-of-the-art medical imaging models
