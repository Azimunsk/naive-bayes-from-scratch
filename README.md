# Naive Bayes from Scratch

This repository contains an implementation of the **Gaussian Naive Bayes classifier from scratch**, written in Python (no `sklearn`). It is tested on:

- 🧪 **Wheat Seed Dataset**
- 🌸 **Iris Dataset**

---

## 🔧 Files

- `wheat_seed_naive_bayes.py`: Classifies wheat seed varieties (3 classes) using 7 numerical features.
- `iris_naive_bayes.py`: Classifies iris flower species using 4 numerical features.
- `wheat-seeds.csv` and `iris.csv`: Sample datasets used for training/testing.

---

## 🚀 How to Run

1. Make sure you have Python 3 and pandas/numpy installed:

   ```bash
   pip install pandas numpy
````

2. Run each script:

   ```bash
   python wheat_seed_naive_bayes.py
   python iris_naive_bayes.py
   ```

---

## ✅ Outputs

Each script prints:

* Number of test samples
* Classification accuracy (usually over 90%)

---

## 📚 Concepts Covered

* Manual calculation of Gaussian probability
* Class-wise mean and variance
* Predicting class by comparing probabilities
* No libraries used for modeling (everything from scratch)

---

## 📌 Note

This is a basic implementation for educational purposes.
No Laplace smoothing, prior probabilities, or vectorized optimizations are applied.
---

