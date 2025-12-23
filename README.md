# Machine-Learning-on-Second-Harmonic-Generation-Microscopy-Images
This repository contains the machine learning pipeline developed for the quantitative analysis of liver collagen remodeling using Second-Harmonic Generation (SHG) microscopy images. The methodology is based on the study titled "Quantitative analysis of arsenic- and sucrose-induced liver collagen remodeling using machine learning on second-harmonic generation microscopy images".

Key Features:

Binary Classification: A neural network model is trained to classify SHG images of rat liver tissue into fibrotic and non-fibrotic categories based on collagen presence.

Feature Selection: Four key statistical features were identified using a Random Forest algorithm for optimal classification:

Percentage of pixels above 15% noise threshold

Mean-to-Standard Deviation ratio (Mean/Std)

Statistical mode (reflecting image noise)

Total intensity sum

Unsupervised Validation: K-Means clustering was applied as a preprocessing step to reduce human bias and validate expert annotations, ensuring robust data grouping before supervised training.

Model Architecture: A simple yet effective binary classifier using a single sigmoid neuron, achieving high accuracy with:

Binary cross-entropy loss

Adam optimizer

20-fold cross-validation

Performance: The model achieved perfect classification metrics:

Accuracy, F1-score, Sensitivity, Specificity = 1.0

Quantitative Output: The pipeline outputs fibrosis risk percentages per dietary group and analyzes collagen fiber orientation using angular distribution histograms.

Dataset:

240 high-resolution SHG images (100×100 pixels)

160 healthy tissue images, 80 fibrotic tissue images

Four dietary groups: Control, Arsenic, Sucrose, Arsenic+Sucrose

Applications:

Early detection of liver fibrosis

Quantitative assessment of collagen remodeling

Dietary risk factor analysis for NAFLD (Non-Alcoholic Fatty Liver Disease)

This repository provides researchers with a reproducible and interpretable ML workflow for automating fibrosis detection from SHG images, combining SHG microscopy with machine learning for biomedical image analysis.
