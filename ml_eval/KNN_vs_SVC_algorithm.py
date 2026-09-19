# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:56:15 2024

@author: Tomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# predict a class of this point -- user defines
point_for_prediction = [4.1, 6.4] 

# load the document
data = np.genfromtxt('Book3.csv', delimiter=',')

# x - data are point values in plane and y - data is label
x = data[:, :2]
y = data[:, 2].astype(np.int64)

# split data in two training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Variables to track best k and gamma
best_k = 1
best_gamma = 0.001
best_C = 0.01
best_accuracy_knn = 0
best_accuracy_svc = 0

# Store results for comparison
results = {"KNN": {}, "SVC": {}}

# KNN algorithm
for k in range(1, 50):  # Test different k values
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_train, y_train)  # Use only training data
    y_pred = clf_knn.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy_knn:
        best_accuracy_knn = accuracy
        best_k = k

# Final KNN with best k
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=best_k)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

# Store KNN results
results["KNN"]["Accuracy"] = accuracy_score(y_test, y_pred_knn)
results["KNN"]["Precision"] = precision_score(y_test, y_pred_knn, average='weighted')
results["KNN"]["Recall"] = recall_score(y_test, y_pred_knn, average='weighted')
results["KNN"]["F1-Score"] = f1_score(y_test, y_pred_knn, average='weighted')

# Plotting decision region
plot_decision_regions(x, y, clf=clf_knn, legend=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN with K=' + str(best_k))
plt.ylim((6, 6.5))
plt.xlim((4.02, 4.23))
plt.show()

print("Best KNN k:", best_k)
print("Accuracy KNN:", best_accuracy_knn)
print("Confusion Matrix KNN:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report KNN:\n", classification_report(y_test, y_pred_knn))


# SVC algorithm
for gamma in np.linspace(0.001, 100, 100):  # Test different gamma values
   for C in np.linspace(0.01, 100, 100):    #Test different C values
        clf_svc = SVC(gamma=gamma, C=C)
        clf_svc.fit(X_train, y_train)
        y_pred = clf_svc.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy_svc:
            best_accuracy_svc = accuracy
            best_gamma = gamma
            best_c = C

# Final SVC with best gamma
clf_svc = SVC(gamma=best_gamma, C=best_c)
clf_svc.fit(X_train, y_train)
y_pred_svc = clf_svc.predict(X_test)

# Store SVC results
results["SVC"]["Accuracy"] = accuracy_score(y_test, y_pred_svc)
results["SVC"]["Precision"] = precision_score(y_test, y_pred_svc, average='weighted')
results["SVC"]["Recall"] = recall_score(y_test, y_pred_svc, average='weighted')
results["SVC"]["F1-Score"] = f1_score(y_test, y_pred_svc, average='weighted')

# Plotting decision region
plot_decision_regions(x, y, clf=clf_svc, legend=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVC with Gamma=' + str(round(best_gamma, 2)) + 'and C=' + str(best_c))
plt.ylim((6, 6.5))
plt.xlim((4.02, 4.23))
plt.show()

print("Best SVC gamma:", best_gamma)
print("Accuracy SVC:", best_accuracy_svc)
print("Confusion Matrix SVC:\n", confusion_matrix(y_test, y_pred_svc))
print("Classification Report SVC:\n", classification_report(y_test, y_pred_svc))

# Compare and Print Results 
print("----- Comparison Results -----")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    knn_value = results["KNN"][metric]
    svc_value = results["SVC"][metric]
    
    if knn_value > svc_value:
        winner = "KNN"
    elif svc_value > knn_value:
        winner = "SVC"
    else:
        winner = "Tie"
    
    print(f"{metric}: KNN = {knn_value:.2f}, SVC = {svc_value:.2f} --> Winner: {winner}")
    
# Predict the new point
print("\n----- Point Label Prediction -----")
prediction_knn = clf_knn.predict([point_for_prediction])  # Wrap the point in a list
print("Predicted KNN for point", point_for_prediction, ":", prediction_knn)

prediction_svc = clf_svc.predict([point_for_prediction])
print("Predicted SVC for point", point_for_prediction, ":", prediction_svc)