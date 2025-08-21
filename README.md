Of course, here's a README file for your project.

-----

# Diabetes Prediction using Support Vector Machine (SVM)

## 📖 Overview

This project implements a machine learning model to predict whether a person has diabetes based on several diagnostic health measurements. The model is built using a **Support Vector Machine (SVM)** classifier with a linear kernel. The code includes steps for data preprocessing, model training, evaluation, and saving the trained model for future use.

-----

## 📂 Project Structure

```
.
├── diabetes.csv          # The dataset used for training and testing
├── train_model.py        # The Python script to train and save the model
├── diabetes_model.pkl    # The saved, trained SVM model object
└── scaler.pkl            # The saved StandardScaler object for data scaling
```

-----

## 💻 Getting Started

### Prerequisites

Make sure you have Python 3 installed on your system. You will also need the following libraries:

  * pandas
  * numpy
  * scikit-learn
  * joblib

### Installation

1.  **Clone the repository** or download the files to your local machine.
2.  **Install the required libraries** using pip:
    ```bash
    pip install pandas numpy scikit-learn joblib
    ```

### Usage

1.  Place the `diabetes.csv` file in the same directory as the Python script.
2.  Run the script from your terminal:
    ```bash
    python train_model.py
    ```

The script will perform the following actions:

  * Load and preprocess the `diabetes.csv` dataset.
  * Scale the features using `StandardScaler`.
  * Split the data into training (80%) and testing (20%) sets.
  * Train an SVM model on the training data.
  * Save the trained model as `diabetes_model.pkl` and the scaler as `scaler.pkl`.
  * Evaluate the model's accuracy on both the test set and the training set and print the results.
  * Demonstrate how to predict the outcome for a new, single data point.

-----

## 🤖 Model Details

  * **Algorithm**: Support Vector Classifier (`sklearn.svm.SVC`) with a `linear` kernel.
  * **Data Preprocessing**: The features are standardized using `StandardScaler` to ensure that all features contribute equally to the model's performance. This is crucial for distance-based algorithms like SVM.
  * **Data Splitting**: The dataset is split using `train_test_split` with `stratify=y` to maintain the same proportion of diabetic and non-diabetic samples in both the train and test sets, which is important for imbalanced datasets.
  * **Evaluation**: The model's performance is measured using the **accuracy score**. The script outputs the accuracy on both the unseen test data and the training data.

-----

## 📈 Example Prediction

The script includes a section to show how to use the trained model and scaler to make a prediction on new data.

**Sample Input Data:**
`(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)`

```python
input_data=(6, 148, 72, 35, 0, 33.6, 0.627, 50)
```

**Prediction Steps:**

1.  The input tuple is converted to a NumPy array.
2.  The array is reshaped to a 2D array, as the model expects a batch of inputs.
3.  The data is scaled using the **same scaler object** (`scaler.pkl`) that was fitted on the training data.
4.  The `model.predict()` method is called on the scaled data to get the final prediction.

**Output:**
The script will print the prediction, where:

  * `1` indicates the person is predicted to have diabetes.
  * `0` indicates the person is predicted to not have diabetes.
