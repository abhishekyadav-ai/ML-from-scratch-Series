## Random Linear Classification

This implementation builds a simple linear classifier from scratch using NumPy, without relying on libraries like sklearn.
 
 The model:
- randomly generates multiple linear decision boundaries
- evaluates each boundary on training data
- selects the one with the lowest classification error

This demonstrates a key idea:
> A reasonable decision boundary can be found through random search, but it is computationally inefficient and not scalable.

---

## Experiment 1: Training on Full Dataset

All data points were used as training data to find the best separating line.

**Result:**
- The model finds a linear boundary that separates most data points.
- Training error is low, but this does not reflect true performance.
- The result can be seen as Dog vs Cat.png

---

## Experiment 2: Train-Test Split (80/20)

The dataset was split into:
- 80% training data  
- 20% testing data  

The model was trained on the training set and evaluated on unseen test data.

**Key Observation:**
- Model performed well even on the testing data
- This highlights the importance of evaluating models on unseen data to avoid misleading results.

---

##Visualization

![Decision Boundary](results.png)

---
##Hyperparameter Tuning using Cross-Validation

To improve model selection, k-fold cross-validation was used to determine the optimal number of random trials (k).

### Method:
- The dataset was split into k folds
- For each value of k (number of random classifiers generated):
  - The model was trained on k-1 folds
  - Evaluated on the remaining fold
- The average error across folds was computed

### Key Insight:
- Increasing the number of random trials improves the chance of finding a better decision boundary
- However, beyond a certain point, improvement slows down while computation cost increases

This demonstrates the trade-off between:
- model performance  
- computational efficiency
------

## Key Takeaways

- Random search can approximate a solution but is inefficient.
- Training accuracy alone is not a reliable metric.
- Proper evaluation requires a train-test split.
- This method serves as a baseline before implementing learning-based algorithms like Perceptron.
