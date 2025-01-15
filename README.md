# CyanoRamanDL

This project combines Raman spectroscopy and deep learning for the classification of toxic cyanobacteria species, enabling early detection and contributing to water quality monitoring and the management of *Harmful Algal Blooms* (HABs).
The main objective of this project is to classify toxic cyanobacteria species such as *Dolichospermum crassum*, *Aphanizomenon sp.*, *Planktothrix agardhii*, and *Microcystis aeruginosa* using deep learning models. Additionally, it aims to enhance model interpretability through SHAP (Shapley Additive Explanations) and provide tools that facilitate both the replication of results and the extension of the analysis to new data.

This repository contains the following scripts:

### `preprocessing_raman_spectra_cyanobacteria.ipynb`
This script provides a complete pipeline for preprocessing Raman spectra. It includes baseline correction using ALS, noise filtering with the Savitzky-Golay method, normalization and standardization of spectral intensities, and the removal of contaminated spectra or those with low signal-to-noise ratios. It produces a clean and homogeneous dataset ready for analysis.

### `outliers_detection_raman_cyanobacteria.ipynb`
This script detects outliers in Raman spectra using the Mahalanobis distance calculated per class. It determines the optimal number of clusters through the elbow method and visualizes outliers on 2D Raman maps. 

### `LDA_visualization_raman_cyanobacteria.ipynb`
This script explores the separability of classes using Linear Discriminant Analysis (LDA). It reduces dimensionality and validates preprocessing consistency by showing the grouping of species in a visualizable 2D.

### `1DCNN_onlyPreprocessed_raman_cyanobacteria.ipynb`
This script trains a 1D convolutional neural network (CNN) using only preprocessed spectra. It includes data preparation by loading preprocessed spectra and dividing them into training and testing sets. The model is defined with an architecture based on convolutional and dense layers. It trains the model with adjustable hyperparameters such as epochs and loss functions, and evaluates results using metrics like accuracy and confusion matrix.

### `multichannel_1DCNN_raman_cyanobacteria.ipynb`
This script implements a multichannel 1D-CNN model that combines raw spectra, baseline estimations, and preprocessed data. Their outputs are fused in dense layers to evaluate performance improvements over the single-channel model. It allows the analysis of how combined information from different channels improves the model's accuracy and generalization.

### `shap_values_raman_cyanobacteria.ipynb`
This script computes SHAP values to analyze model interpretability. It identifies key spectral regions contributing to classification, showing their importance globally and for specific classes. It includes detailed visualizations of spectral band contributions to model performance.

### `validate_culture_medium_influence.ipynb`
This script evaluates the impact of the culture medium (BG11 and BG110) on spectra and classification. It compares spectra obtained in different media and quantitatively analyzes differences using CNN and LDA models. It also includes visual comparisons to identify how culture conditions affect relevant bands.

### `validation_DLmodels_withRamanImages.ipynb`
This script performs cross-validations with independent data. It predicts on mixed cyanobacteria samples and generates 2D Raman maps to analyze classification patterns. 
