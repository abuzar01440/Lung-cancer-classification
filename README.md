# Lung Cancer Classification Using ConvNeXtTiny with Transfer Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15256078.svg)](https://doi.org/10.5281/zenodo.15256078)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository provides the open-source implementation of the research paper **"Lung Cancer Classification by Using Transfer Learning"**, published in the *Spectrum of Engineering Sciences*. The study leverages the lightweight **ConvNeXtTiny** model to classify lung cancer histopathological images into three categories: **adenocarcinoma**, **squamous cell carcinoma**, and **benign** lesions, achieving high accuracy with low computational costs.

---

## 📌 Abstract
Lung cancer remains a leading cause of cancer-related mortality worldwide. This work addresses the challenge of deploying deep learning models in resource-constrained medical settings by adapting the computationally efficient **ConvNeXtTiny** architecture. Using transfer learning, the model achieves **98.31% test accuracy**, **98.31% precision**, **98.31% recall**, and **98.43% F1-score** on a curated dataset of histopathological images. The results highlight its potential for practical clinical applications, particularly in environments with limited computational resources.



## 🛠️ Installation
### Dependencies
- Python 3.10+
- TensorFlow 2.10+
- Keras
- scikit-learn
- NumPy
- Pandas
- Matplotlib


---

## 🗂️ Dataset
The **Lung Cancer Histopathological Images** dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images/data). It contains three classes:
1. **Adenocarcinoma** (glandular tissue origin)
2. **Squamous Cell Carcinoma** (squamous cell origin)
3. **Benign** (non-cancerous tissue)

### Preprocessing
- **Split**: 90% training, 10% testing (stratified sampling).
- **Augmentation** (applied to training data):
  - Rotation: 120°
  - Width/Height Shift: 0.12
  - Shear/Zoom: 0.12
  - Horizontal/Vertical Flip
  - Brightness Range: [0.5, 2.5]
  - Resize: 256x256 pixels
  - Rescaling: 1/255.0

---

## 🧠 Model Architecture
The **ConvNeXtTiny** model is adapted using transfer learning:
1. **Base Model**: Pre-trained on ImageNet (`include_top=False`).
2. **Custom Layers**:
   - Global Average Pooling
   - Flatten Layer
   - Dense (1234 units, ELU activation)
   - Dense (1024 units, ELU activation)
   - Output Layer (3 units, Softmax activation)

### Training Configuration
- **Optimizer**: Adam (`lr=0.0001`)
- **Loss**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: ModelCheckpoint (saves best model)

---


## 📊 Results
| Metric               | Training Data | Testing Data |
|----------------------|---------------|--------------|
| **Accuracy**         | 98.47%        | 98.31%       |
| **Precision**        | 98.47%        | 98.31%       |
| **Recall**           | 98.47%        | 98.31%       |
| **F1-Score**         | 98.47%        | 98.43%       |
| **Cross-Entropy Loss** | 0.0375       | 0.0572       |

---

## 📜 Citation
If you use this code or findings in your work, please cite the original paper:
```bibtex
@article{maqbool2025lung,
  title={Lung Cancer Classification by Using Transfer Learning},
  author={Maqbool, Muhammad Hashim and Shahid, Abuzar and Mumtaz, Gohar},
  journal={Spectrum of Engineering Sciences},
  volume={3},
  number={4},
  pages={191--199},
  year={2025},
  doi={10.5281/zenodo.15256078}
}
```

---

## 📞 Contact
For questions or collaborations, contact:
- Muhammad Hashim Maqbool: [hashimmaqbool143@gmail.com](mailto:hashimmaqbool143@gmail.com)
- Abuzar Shahid: [abuzarbhutta@gmail.com](mailto:abuzarbhutta@gmail.com)
- Gohar Mumtaz: [ghor.m@superior.edu.pk](mailto:ghor.m@superior.edu.pk)

---

## 🔗 Links
- [Research Paper](https://sesjournal.com/index.php/1/article/view/246/264)
- [Dataset (Kaggle)](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images/data)
- [DOI](https://doi.org/10.5281/zenodo.15256078)
```
