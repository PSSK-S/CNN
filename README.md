# 🐱🐶 CNN Image Classifier — Cats vs Dogs
A Convolutional Neural Network (CNN) built using **TensorFlow and Keras** to classify images of cats and dogs.  
This project follows the classic *Deep Learning A–Z* structure, modernized for **Colab (GPU runtime)** and **TensorFlow 2.17 / Keras 3** compatibility.  

## 🚀 Overview
The model automatically learns image features through convolution and pooling layers to distinguish between two classes
It includes preprocessing, model building, training, and prediction stages.

## 🧠 Features
- Image preprocessing with `ImageDataGenerator`  
- Real-time data augmentation (shear, zoom, flip)  
- Convolution + MaxPooling + Flatten + Dense layers  
- Binary classification using **sigmoid activation**  
- GPU-accelerated training on Google Colab (T4 GPU)  
- Single-image prediction support  

## 🧩 Folder Structure
```
dataset/
 ├── training_set/
 │   ├── cats/
 │   └── dogs/
 ├── test_set/
 │   ├── cats/
 │   └── dogs/
```

## ⚙️ Tech Stack
- **Language:** Python 3  
- **Framework:** TensorFlow 2.17 / Keras 3  
- **Environment:** Google Colab (GPU)  
- **Dataset:** Cats vs Dogs (custom local copy)

## 🧾 Model Summary
- **Input:** 64×64 RGB images  
- **Architecture:**
  - 2 × Conv2D + MaxPooling2D layers  
  - Flatten layer  
  - Dense(128, relu)  
  - Dense(1, sigmoid)  
- **Optimizer:** Adam  
- **Loss:** Binary Crossentropy  
- **Metric:** Accuracy  

## 📈 Training Example
| Metric | Value (approx.) |
|:-------|:----------------|
| Accuracy | 85–90 % |
| Loss | ↓ steadily across epochs |
| Runtime | ~25 s / epoch (T4 GPU) |

## 🧪 Single Prediction
```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
result = cnn.predict(test_img)
print("Dog" if result[0][0] > 0.5 else "Cat")
```

## 📂 How to Run
1. Mount Google Drive and copy dataset:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp -r "/content/drive/My Drive/dataset" /content/
   ```
2. Enable GPU:  
   `Runtime → Change runtime type → T4 GPU`
3. Run all notebook cells.

## 🧾 Author
**Satya Sai Kiran Pithani**  
> CNN implementation practice project — part of Deep Learning A–Z course.
