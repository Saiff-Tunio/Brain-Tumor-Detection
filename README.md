# 🧠 Brain Tumor Detection Using Deep Learning

This project focuses on the detection of brain tumors from MRI images using various state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformer-inspired architectures. Models are trained using the **Kaggle Brain Tumor MRI Dataset**, consisting of 7023 labeled MRI images.

All models are implemented in **PyTorch**, with comparison between their **initial (frozen)** and **fine-tuned** versions for deeper performance analysis.

---

## 📂 Dataset

* **Source**: [Kaggle – Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Total Images**: 7023
* **Classes**:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor

---

## 🏗️ Model Architectures

Each model was tested in two configurations:

* **Initial**: Only the classifier head is trained (feature extractor is frozen).
* **Fine-Tuned**: Selected backbone layers are unfrozen and trained for deeper learning.

### 🧠 CNN & ViT-based Architectures

| Model Name     | Initial Accuracy       | Fine-Tuned Accuracy    |
| -------------- | ---------------------- | ---------------------- |
| ResNet18       | 92.68%                 | 98.70%                 |
| ResNet50       | 90.85%                 | 98.63%                 |
| DenseNet121    | 94.74%                 | 96.64%                 |
| EfficientNetB0 | 97.03%                 | 97.73%                 |
| InceptionV3    | 92.14%                 | 92.68%                 |
| MobileNetV2    | 92.98%                 | 94.43%                 |
| ConvNeXtBase   | 96.19%                 | 97.71%                 |

> 📊 *Accuracy, classification reports, and confusion matrices for each model are provided in the notebooks.*

---

## 🧪 Evaluation

Each model is evaluated using:

* Test Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-Score)
* Visualizations of predictions

Early stopping and learning rate adjustment techniques are used for robust training.

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/Saiff-Tunio/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. Open the `.ipynb` files in Jupyter Notebook or VS Code.

3. Run each notebook individually to train and evaluate the models.

---

## 📌 Project Highlights

* Transfer learning with multiple architectures
* Manual fine-tuning of backbone layers
* Extensive training and evaluation pipeline
* GPU support with automatic device handling
* Visual output for model predictions

---

## 📁 Repository Structure

```
brain-tumor-detection/
│
├── ConvNeXtBase – Initial vs. Fine-Tuned Model.ipynb
├── DenseNet121 – Initial vs. Fine-Tuned Model.ipynb
├── EfficientNetB0 – Initial vs. Fine-Tuned Model.ipynb
├── InceptionV3 – Initial vs. Fine-Tuned Model.ipynb
├── MobileNetV2 – Initial vs. Fine-Tuned Model.ipynb
├── ResNet18 – Initial vs. Fine-Tuned Model.ipynb
├── ResNet50 – Initial vs. Fine-Tuned Model.ipynb
└── README.md
```
---

## 🛠️ Libraries Used

* [`torch`](https://pytorch.org/) - Core deep learning framework
* [`torchvision`](https://pytorch.org/vision/stable/index.html) - Pre-trained models and data utilities
* `numpy` - Tensor manipulation and statistics
* `matplotlib` - Image and metric visualization
* `seaborn` - Enhanced plotting for confusion matrices
* `scikit-learn` - Metrics: classification report and confusion matrix
* `kaggle` - Accessing datasets directly from KaggleHub

---

## 📊 Results

The best-performing model in this experiment was **Fine-Tuned ResNet18** achieving **98.70% test accuracy**. Performance improved significantly after fine-tuning in most architectures.

---

## 🔭 Future Work

* **Training on Diverse Datasets:** Extend model evaluation by training and validating on additional brain tumor MRI datasets (e.g., BraTS, Figshare) to assess generalization and robustness across institutions.
* **Integration of Vision Transformers (ViT):** Incorporate transformer-based models for potentially better feature learning and global attention in medical image analysis.
* **Explainable AI (XAI):** Implement interpretability methods such as Grad-CAM, SHAP, or LIME to visualize and explain model predictions for clinical trust.
* **Web Deployment:** Build a user-friendly interface using Streamlit or Flask for real-time brain tumor detection from uploaded MRI scans.
* **Multimodal Analysis:** Combine MRI imaging data with clinical metadata for richer, more accurate predictions.

---

## 👨‍💻 Author

**Saifullah**
📚 *Student, Machine Learning & AI Enthusiast*
📬 \[saifullah.tunio1@gmail.com]
🧠 Passionate about medical imaging, computer vision, and model interpretability.

---


