# Skin-cancer-classification
Multi-modal classification system for dermoscopic skin lesion diagnosis using both image and clinical metadata features.

## Dataset
- **Source**: HAM10000 (Kaggle Skin Cancer MNIST)
- **Size**: 10,015 dermoscopic images
- **Classes**: 7 skin lesion types (nv, mel, bkl, bcc, akiec, vasc, df)
- **Modalities**: RGB images + clinical metadata (age, sex, localization, dx_type)
- **Challenge**: Severe class imbalance (67% nv, <2% minority classes)

## Approach

### Multi-Modal Architecture
- **Image Pipeline**: CNN feature extraction with transfer learning
- **Tabular Pipeline**: Clinical metadata preprocessing with encoding/scaling
- **Fusion Strategy**: Concatenated feature vectors for combined predictions

### Models Implemented

#### Traditional ML
- **Naive Bayes**: GaussianNB with PCA-reduced image features
- **K-Nearest Neighbors**: k=10 with standardized multi-modal features
- **Decision Tree**: Max depth 15 with entropy splitting criterion
- **XGBoost**: Gradient boosting with class weight balancing

#### Deep Learning
- **CNN**: Custom architecture with data augmentation and dropout
- **MLP**: 3-layer network (512-256-128) processing flattened images + metadata

### Feature Engineering
- **Images**: Resized to 64x64, normalized, PCA reduction to 100 components
- **Metadata**: One-hot encoding for categorical, standardization for numerical
- **Combined**: Horizontal concatenation creating 123-dimensional feature vectors

## Results
- **Best Model**: MLP Combined - 76.7% accuracy, 0.75 weighted F1-score
- **Traditional ML**: Decision Tree 72.4% (tabular), KNN 70.7% (combined)
- **Deep Learning**: CNN 57% accuracy (struggled with class imbalance)
- **Class Performance**: Strong on majority class (nv: F1=0.90), weak on rare classes

## Key Challenges
- Severe class imbalance requiring weighted loss and oversampling
- Image quality inconsistencies (lighting, artifacts, resolution)
- High-dimensional feature space requiring dimensionality reduction

## Technologies
- Python, scikit-learn, TensorFlow/Keras
- PIL for image processing, PCA for dimensionality reduction
- Pandas, NumPy for data manipulation
- XGBoost for gradient boosting

