# SMILES - Smile Detection using Neural Networks

This repository contains a deep learning project focused on classifying human facial images as either "smiling" or "not smiling" using the **SMILES** dataset[cite: 1, 2]. The project explores and compares the capabilities of **Fully Connected Feed Forward Neural Networks (FFNN)** and **Convolutional Neural Networks (CNN)**, along with a systematic hyperparameter optimization process[cite: 1, 2].

## 📊 Dataset & Preprocessing
* **Dataset:** The SMILES database consists of 13,165 grayscale facial images categorized into positive (smiling) and negative (not smiling) classes[cite: 1].
* **Data Splitting:** The dataset was split into 80% for training and 20% for validation[cite: 2].
* **Normalization:** Pixel values were normalized between 0 and 1 using `Rescaling(1./255)` to ensure stable and faster gradient convergence[cite: 2].
* **Data Optimization:** Implemented TensorFlow's `tf.data` API pipeline incorporating `cache()`, `shuffle(1000)`, and `prefetch(buffer_size=AUTOTUNE)` to maximize training performance and eliminate disk I/O bottlenecks[cite: 2].

## 🧠 Architectures & Experimental Analysis

### 1. Fully Connected Feed Forward Neural Network (FFNN / MLP)
* **Baseline:** A shallow network architecture flattening the spatial input into a hidden layer with 128 neurons[cite: 2]. The initial baseline model exhibited high instability and significant accuracy fluctuations during validation[cite: 2].
* **Optimizations:** Explored changes in learning rates, varying hidden layer neuron counts, and switching the output layer activation function from Softmax to **Sigmoid**[cite: 2]. Utilizing a Sigmoid activation vastly stabilized the evaluation metrics for this binary classification setup[cite: 2].

### 2. Convolutional Neural Network (CNN) - *Best Performer*
* **Baseline:** Started with a simple architecture utilizing a single `Conv2D` layer (16 filters, 3x3 kernel) followed by `MaxPooling2D`[cite: 2]. The baseline fell short of capturing advanced features within lower epoch limits[cite: 2].
* **Systematic Enhancements:**
  * **Training Extension:** Increased training duration from 5 to 10 epochs[cite: 2].
  * **Architecture Depth & Regularization:** Added consecutive convolution blocks scaling up to 32 and 64 filters to learn granular hierarchical features[cite: 2]. Integrated a **Dropout layer (0.3)** to penalize overfitting[cite: 2].
  * **Hyperparameter Tuning:** Reduced kernel dimensions and adjusted the optimization process using the Adam optimizer with `learning_rate=0.0001`[cite: 2].
* **Outcome:** The optimized CNN model achieved a top **Validation Accuracy of 93.20%**, demonstrating reliable generalization without falling into overfitting pitfalls[cite: 2].

## 📈 Evaluation & Summary
Due to their inherent capacity to preserve and learn local spatial dependencies in imagery, **CNNs significantly outperformed the FFNN architecture**[cite: 2]. Because the FFNN flattens spatial matrices into 1D vectors, it inevitably strips away pixel neighborhood correlations, limiting its overall accuracy despite heavy optimization[cite: 2]. The final CNN structure, backed by an optimized learning rate and proper regularization, demonstrated stable and elegant convergence[cite: 2].

## 🛠️ Tech Stack
* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab

## 👥 Developers
This project was developed as a collaborative academic study at Bursa Teknik University, Department of Mechatronics Engineering[cite: 2]:
* Samet Ozan Topcu[cite: 2]
* İbrahim İdris İbrahim[cite: 2]
