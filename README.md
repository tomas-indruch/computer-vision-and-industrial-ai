# Industrial Machine Learning & Computer Vision Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM%20%7C%20KNN-orange)
![Optimization](https://img.shields.io/badge/Optimization-Genetic%20Algorithms-brightgreen)

A multi-disciplinary repository covering machine learning classification, gradient-based feature extraction for industrial inspection, and evolutionary parameter optimization in manufacturing processes.

---

## 📌 Modules Overview

### 1. Parametric Optimization (Genetic Algorithms)
* Applied evolutionary search to multi-objective CNC machining constraints (feed rate, cutting speed, tool wear, and operational cost).
* Automated parameter evaluation balancing material removal rate against thermal and roughness tolerances[cite: 1].

### 2. Pattern Classification: KNN vs. Support Vector Classifier (SVC)
* Evaluated non-linear decision boundaries using hyperparameter grid tuning ($C$, $\gamma$ for RBF kernels vs. $k$-neighbors)[cite: 1].
* Benchmarked models across Precision, Recall, and F1-score metrics on multi-class experimental datasets, achieving up to 96% accuracy[cite: 1].

### 3. Industrial Computer Vision (OpenCV)
* Gradient-based inspection pipeline implementing tuned Gaussian filtering, directional Sobel derivatives, and double-threshold Canny edge extraction[cite: 1].
* Topological hierarchy extraction via `cv2.findContours` for automated geometric verification[cite: 1].

---
