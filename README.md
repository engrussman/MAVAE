# **Brain Age Estimation using Multimodal MRI Data with Multi-Task Adversarial Variational Autoencoder (M-AVAE)**

This project focuses on predicting biological brain age using multimodal MRI data (sMRI and fMRI) while also incorporating gender prediction as an auxiliary task. We introduce a novel Multi-Task Adversarial Variational Autoencoder (M-AVAE), which leverages adversarial and variational learning techniques to disentangle shared and unique information from multimodal data, enhancing the accuracy and robustness of brain age estimation.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Comparison](#performance-comparison)
- [License](#license)
- [Contact](#contact)

---

## **Introduction**

Accurate estimation of biological age (BA) from neuroimaging data is crucial for understanding brain development and detecting neurodegenerative diseases. BA provides insights into individual differences in age-related traits, often more effectively than chronological age (CA), especially when studying conditions like Alzheimer's, HIV, and traumatic brain injuries. While non-imaging approaches for BA estimation have been explored extensively, they lack specificity to particular organs, like the brain.

Our study addresses this gap by focusing on the integration of functional magnetic resonance imaging (fMRI) and structural magnetic resonance imaging (sMRI) to estimate brain age. Additionally, we incorporate gender prediction, as gender differences significantly influence the brain aging process. The novel M-AVAE framework introduced in this project disentangles the shared and unique features of the sMRI and fMRI data to improve the estimation of brain age.
![Brain Scan Visualizations](images/visual_sMRI_fMRI.png)
*Visualization of sMRI and fMRI brain scans across different age groups and sex.*

---

## **Methodology**

This project introduces the **Multi-Task Adversarial Variational Autoencoder (M-AVAE)**, which integrates both adversarial and variational autoencoders to perform brain age estimation and gender prediction. The core innovation lies in disentangling the shared and unique components of the latent features extracted from sMRI and fMRI data.

- **Feature Extraction**: Initial features are selected using a Random Forest-based filter method. This reduces the dimensionality of the data, making it feasible to process.
  
- **Multi-Task Learning**: The model is designed to perform two tasks simultaneously: brain age estimation and gender prediction. By leveraging multitask learning, the model captures gender-specific aging patterns.
  
- **Latent Space Disentanglement**: Latent variables are divided into shared and unique components across the two imaging modalities (sMRI and fMRI). The adversarial and variational losses ensure that the shared information is robust, while modality-specific information is disentangled effectively.
  
- **Loss Functions**: The model optimizes a combination of adversarial, variational, regression, and classification losses to ensure both accurate age estimation and reliable gender classification.

---

## **Results**

Extensive experimentation on publicly available neuroimaging datasets demonstrated the superior performance of the proposed M-AVAE model over several baseline methods. Key results include:

- **Performance Metrics**: The M-AVAE outperforms traditional methods such as Random Forest, Support Vector Regression, and AAE models in terms of Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Pearson Correlation Coefficient (PCC).
  
- **Multimodal Fusion**: Incorporating both sMRI and fMRI data significantly improved the model’s performance, with the M-AVAE showing the lowest MAE in the age estimation task.

- **Gender Prediction**: By incorporating gender as an auxiliary task, the model demonstrated enhanced accuracy, leveraging the biological differences in brain aging patterns between males and females.

Below are sample performance comparisons of M-AVAE with other methods:

| **Method**   | **MAE**   | **RMSE**  | **PCC**   |
|--------------|-----------|-----------|-----------|
| Random Forest (RF)   | 4.96 ± 3.00 | 5.80 ± 4.27 | 0.65 ± 0.29 |
| SVR   | 4.46 ± 2.82 | 5.27 ± 4.04 | 0.69 ± 0.25 |
| **M-AVAE (Proposed)**  | **2.77 ± 1.57** | **3.18 ± 1.90** | **0.82 ± 0.13** |

The M-AVAE model provides superior accuracy and demonstrates its robustness across different age groups and imaging modalities.

---

## **Model Architecture**

![M-AVAE Architecture](images/model.png)

The architecture of the M-AVAE consists of two encoders, each handling one modality (sMRI or fMRI). These encoders generate a latent space, which is divided into shared and unique parts. The decoder reconstructs the original inputs, and multitask learning is used to predict both the age and gender of the subjects.

- **Encoders**: For each modality (sMRI and fMRI), a dedicated encoder extracts relevant features and divides them into shared and unique components.
- **Decoder**: The decoder reconstructs the inputs using both the shared and unique latent features.
- **Adversarial and Variational Losses**: These losses ensure that the shared and unique latent spaces are correctly disentangled, improving the model's robustness.

---

## **Performance Comparison**

The performance of the proposed M-AVAE model was evaluated against various state-of-the-art methods, showing its superiority, particularly in the integration of multimodal data and the inclusion of gender prediction.

| **Model**                  | **MAE** | **RMSE** | **PCC** |
|----------------------------|---------|----------|---------|
| 3D-Peng (2021)              | 4.17    | 5.37     | —       |
| Age-Net-Gender (2021)       | 3.61    | 4.76     | —       |
| CAE (2023)                  | 2.71    | 3.68     | 0.87    |
| **Proposed M-AVAE**         | **2.77** | **3.18** | **0.82** |

---

## **Usage**

### **1. Setup and Installation**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/brain-age-prediction.git
    cd brain-age-prediction
    ```

2. **Install dependencies**:
    Ensure you have Python 3.8 or above installed. Use the following command to install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

---

### **2. Prepare Your Data**

The model expects **two sets of features** (likely from structural MRI (sMRI) and functional MRI (fMRI)) in the form of NumPy arrays. Here's a guideline for preparing your data:

1. **Convert your brain scans into NumPy files**:
   - Ensure that your MRI scans are preprocessed and saved as `.npy` files (NumPy arrays). Each file should contain features extracted from the scans (e.g., after applying preprocessing, such as resizing, normalization, etc.).
   - If you are starting with medical image files like **NIfTI (.nii)** or **DICOM (.dcm)**, convert them into NumPy arrays using tools like `nibabel` (for `.nii` files) or `pydicom` (for `.dcm` files). Example:

     ```python
     import nibabel as nib
     import numpy as np

     # Load NIfTI image
     img = nib.load('your_scan.nii')
     img_data = img.get_fdata()

     # Save as NumPy array
     np.save('features1.npy', img_data)
     ```

2. **Save the NumPy arrays**:
   - You need to save two sets of features as NumPy files:
     - `features1.npy`: The first set of features (e.g., from sMRI scans).
     - `features2.npy`: The second set of features (e.g., from fMRI scans).
   - Corresponding labels should be saved as:
     - `labels_Age.npy`: Array of the subjects' chronological ages.
     - `labels_Gender.npy`: Array of the subjects' genders (e.g., 0 for male, 1 for female).

---

### **3. Training the Model**

1. **Prepare Training Data**:
   Ensure the following NumPy files exist in the root directory:
   - `features1.npy`: First modality's features (e.g., sMRI data).
   - `features2.npy`: Second modality's features (e.g., fMRI data).
   - `labels_Age.npy`: Ground truth ages.
   - `labels_Gender.npy`: Ground truth genders.

2. **Modify Hyperparameters (Optional)**:
   - You can modify training hyperparameters like `batch_size`, `learning rate`, and `number of epochs` by editing the `train.py` file.

3. **Run the Training Script**:
    ```bash
    python train.py
    ```

   The model will train using the provided datasets and automatically save checkpoints and performance logs.

---

### **4. Testing the Model**

1. **Prepare Testing Data**:
   Convert your test brain scans (sMRI and fMRI) into NumPy arrays similar to the training data.
   - `Test_features1.npy`: First modality features (sMRI).
   - `Test_features2.npy`: Second modality features (fMRI).
   - `Test_labels_Age.npy`: Ground truth age labels for test data.
   - `Test_labels_Gender.npy`: Ground truth gender labels for test data.

2. **Run the Testing Script**:
    ```bash
    python test.py
    ```

   This will run the trained model on the test data, output the predictions, and print evaluation metrics (e.g., accuracy, F1 score, precision, recall).

---

### **5. Inference with New Data**

If you have new data (sMRI and fMRI scans) and want to perform inference (predict age and gender):

1. **Prepare New Data**:
   Ensure your new data is preprocessed and converted into NumPy arrays (`feature1.npy` and `feature2.npy`).

2. **Run Inference**:
    ```bash
    python main.py feature1.npy feature2.npy
    ```

   This will load the model and print predicted age and gender to the console.

---

### **6. Expected Output**

- **Predicted Age**: The model will output the predicted biological age.
- **Predicted Gender**: The model will output the predicted gender (e.g., 0 for male, 1 for female).
  
Example console output:



