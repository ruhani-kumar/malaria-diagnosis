# Malaria Detection System Using Deep Learning
A comprehensive deep learning solution for automated malaria parasite detection from cell microscopy images. This project implements and compares multiple CNN architectures, explores model explainability through Grad-CAM, and estimates prediction confidence using Monte Carlo Dropout. An interactive Streamlit dashboard provides real-time predictions with uncertainty quantification.

Research Focus: Model comparison, transfer learning, explainability (XAI), and uncertainty estimation in medical imaging.

Features

1. Multiple Model Architectures: Baseline LeNet with Batch Normalization, transfer learning with MobileNetV2, and fine-tuned MobileNetV2
2. Comprehensive Model Comparison: Detailed accuracy and loss analysis across training, validation, and test sets
3. Explainability (Grad-CAM): Visual heatmaps showing which regions of cell images influence model predictions
4. Uncertainty Quantification: Monte Carlo Dropout for estimating prediction confidence and model uncertainty
5. Interactive Dashboard: Streamlit-based web application for model comparison, visualization, and inference
6. Real-Time Predictions: Upload images and receive predictions with confidence scores
7. Binary Classification: Parasitized vs. Uninfected cell classification

Model Performance Comparison

<img width="806" height="194" alt="image" src="https://github.com/user-attachments/assets/54b2d1a1-923b-49f2-8ea5-81aa74a47250" />
