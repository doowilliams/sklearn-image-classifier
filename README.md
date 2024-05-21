# sklearn-image-classifier

# Digit Classification with SVM and PCA

This project demonstrates a complete pipeline for digit classification using the Support Vector Machine (SVM) algorithm with Principal Component Analysis (PCA) for dimensionality reduction. The digits dataset from Scikit-learn is used for this purpose.

## Installation

To run this code, you need to have the following Python packages installed:
- numpy
- matplotlib
- scikit-learn

You can install these packages using pip:

```sh
pip install numpy matplotlib scikit-learn
```

## Usage

1. **Load the digits dataset:**

   The digits dataset is loaded from Scikit-learn. This dataset contains 8x8 images of handwritten digits.

2. **Display the first image from the dataset:**

   The first image in the dataset is displayed using Matplotlib to give an idea of what the dataset looks like.

3. **Split the dataset into training and testing sets:**

   The dataset is split into training and testing sets using an 70-30 split.

4. **Preprocess the data:**

   The data is standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.

5. **Dimensionality reduction:**

   PCA is applied to reduce the dimensionality of the dataset to 40 components. This helps in speeding up the training process and can improve the performance of the model.

6. **Define and train the SVM model:**

   An SVM with a linear kernel is defined and trained on the training set.

7. **Make predictions:**

   Predictions are made on the test set using the trained model.

8. **Evaluate the model:**

   The model's performance is evaluated using a confusion matrix and a classification report.

## Code

Here is the complete code for this pipeline:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Display the first image from the dataset
plt.gray()
plt.matshow(digits.images[0])
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality reduction
pca = PCA(n_components=40)  # Reduce to 40 dimensions for example
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define the SVM model
clf = SVC(kernel='linear', random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Explanation

- **Loading the dataset:** The digits dataset is loaded into `X` (features) and `y` (labels).
- **Displaying an image:** The first digit image in the dataset is displayed using Matplotlib.
- **Splitting the dataset:** The dataset is divided into training and testing sets to evaluate the model's performance on unseen data.
- **Preprocessing:** The features are scaled to standardize the dataset, which helps in improving the model's performance.
- **PCA:** Dimensionality reduction is performed to reduce the number of features while retaining most of the variance in the dataset.
- **SVM Model:** A Support Vector Machine with a linear kernel is trained on the training set.
- **Predictions and Evaluation:** The trained model makes predictions on the test set, and its performance is evaluated using a confusion matrix and a classification report.

## Conclusion

This project demonstrates how to build a machine learning pipeline for digit classification using PCA and SVM. The steps include loading the data, preprocessing, dimensionality reduction, training the model, making predictions, and evaluating the model's performance. This approach can be extended to other classification tasks with similar structured datasets.
