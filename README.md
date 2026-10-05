# Financial Modeling and Image Colorization

A machine learning project composed of two independent tasks:

1. **Quantitative Financial Alpha Modeling**
2. **Neural Network Image Colorization**

The project combines model development, evaluation, optimization, and validation across two very different machine learning applications.

---

## Part 1 – Quantitative Financial Alpha Modeling

The first part focuses on building and evaluating multiple financial alpha models.

The workflow includes feature engineering, model training, validation, performance filtering, and combining several alpha signals into a final set of strategies.

### Main Steps

- Financial feature engineering
- Construction and training of multiple alpha models
- Evaluation using financial performance metrics
- Sharpe ratio analysis
- Turnover analysis
- Drawdown and PnL evaluation
- Correlation analysis between alpha strategies
- Selection of valid and sufficiently diverse alphas
- Combination of multiple alpha signals

### Final Results

The final pipeline selected **10 alpha models** that satisfied the required validation criteria.

Example individual Sharpe ratios included:

- `alpha_26`: approximately **3.22**
- `alpha_50`: approximately **2.84**
- `neg_alpha_42`: approximately **2.69**
- `neg_alpha_03`: approximately **2.63**

All 10 selected strategies passed the final Sharpe and turnover checks.

The project also analyzes correlations between the selected strategies in order to avoid relying only on highly similar alpha signals.

### Supporting Code

`financial_toolkit.py` contains utility functions used by the financial modeling notebook.

---

## Part 2 – Image Colorization

The second part uses a convolutional neural network to transform grayscale images into color images.

The model was implemented using PyTorch and trained on images from the **Caltech101** dataset.

### Dataset

The experiment used:

- **8,677 images**
- **6,941 training images**
- **1,736 validation images**

### Model

The colorization network is built using convolutional blocks and learns to predict RGB image values from grayscale input images.

The workflow includes:

- Image preprocessing
- Conversion from RGB to grayscale
- Train/validation splitting
- Convolutional neural network design
- Model training
- Validation monitoring
- Saving the best-performing model
- Image reconstruction and visualization

### Evaluation

The final model was evaluated using:

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)

Final validation performance:

- **Validation MSE:** approximately `0.00765`
- **Validation PSNR:** approximately `21.17 dB`

The notebook also includes visual comparisons between grayscale input images, predicted colorized images, and original RGB images.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Torchvision
- Google Colab

---

## Repository Files

`financial_alpha_modeling.ipynb` – financial feature engineering, alpha model training, evaluation, and strategy selection.

`financial_toolkit.py` – supporting utilities for the financial modeling pipeline.

`image_colorization.ipynb` – convolutional neural network for grayscale-to-color image reconstruction.

---

## Key Concepts

This project demonstrates experience with:

- Machine learning model development
- Neural networks
- Financial modeling
- Feature engineering
- Performance evaluation
- Sharpe ratio and turnover
- Strategy diversification
- Convolutional neural networks
- Image preprocessing
- Image reconstruction
- Model validation
