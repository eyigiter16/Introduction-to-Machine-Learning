#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# In[2]:


np.random.seed(421)

means = np.array([[+0.0, +2.5],
                  [-2.5, -2.5],
                  [+2.5, -2.0]])

covariances = np.array([
    [
        [+3.2, +0.0],
        [+0.0, +1.2]
    ],
    [
        [+1.2, -0.8],
        [-0.8, +1.2]
    ],
    [
        [+1.2, +0.8],
        [+0.8, +1.2]
    ]])

sizes = np.array([120, 90, 90])

# In[3]:


points1 = np.random.multivariate_normal(means[0, :], covariances[0, :, :], sizes[0])
points2 = np.random.multivariate_normal(means[1, :], covariances[1, :, :], sizes[1])
points3 = np.random.multivariate_normal(means[2, :], covariances[2, :, :], sizes[2])
x = np.vstack((points1, points2, points3))

# In[4]:


# class labels
y = np.concatenate((np.repeat(1, sizes[0]), np.repeat(2, sizes[1]), np.repeat(3, sizes[2])))
k = 3

# In[5]:


plt.figure(figsize=(6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# In[6]:


sample_means = [np.mean(x[y == (c + 1)], axis=0) for c in range(k)]
print("sample means")
print(sample_means)

# In[7]:


sample_covariances = [
    (np.matmul(np.transpose(x[y == (c + 1)] - sample_means[c]), (x[y == (c + 1)] - sample_means[c])) / sizes[c]) for c
    in range(k)]
print("sample covariances")
print(sample_covariances)

# In[8]:


class_priors = [np.mean(y == (c + 1)) for c in range(k)]
print("prior probabilities")
print(class_priors)

# In[9]:


Wcs = np.array([np.linalg.inv(sample_covariances[c]) / -2 for c in range(k)])
wcs = np.array([np.matmul(np.linalg.inv(sample_covariances[c]), sample_means[c]) for c in range(k)])
wc0s = np.array([-(np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])),
                             sample_means[c])) / 2 - np.log(np.linalg.det(sample_covariances[c])) / 2 + np.log(
    class_priors[c]) for c in range(k)])
print("Wcs")
print(Wcs)
print("wcs")
print(wcs)
print("wc0s")
print(wc0s)


# In[10]:


def calc_scores(x):
    scores = np.array([0, 0, 0])
    for i in range(k):
        score = np.matmul(np.matmul(np.transpose(x), Wcs[i]), x)
        score = score + np.matmul(np.transpose(wcs[i]), x)
        score = score + wc0s[i]
        scores[i] = score
    return scores


# In[11]:


y_truth = y
y_predicted = [calc_scores(x[i]) for i in range(len(x))]

# In[12]:


y_pred = np.argmax(y_predicted, axis=1) + 1

# In[13]:


confusion_matrix = pd.crosstab(y_pred, y_truth, rownames=['y_pred'], colnames=['y_truth'])

# In[14]:

print("confusion matrix")
print(confusion_matrix)

# In[15]:


x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), k))

for c in range(k):
    discriminant_values[:, :, c] = (Wcs[c, 0, 0] * x1_grid ** 2) + (Wcs[c, 0, 1] * x1_grid * x2_grid) + (
                wcs[1, 0] * x2_grid * x1_grid) + (wcs[1, 1] * x2_grid ** 2) + (wcs[c, 0] * x1_grid) + (
                                               wcs[c, 1] * x2_grid) + wc0s[c]

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C

plt.figure(figsize=(10, 10))
plt.plot(x[y_truth == 1, 0], x[y_truth == 1, 1], "r.", markersize=10)
plt.plot(x[y_truth == 2, 0], x[y_truth == 2, 1], "g.", markersize=10)
plt.plot(x[y_truth == 3, 0], x[y_truth == 3, 1], "b.", markersize=10)

plt.plot(x[y_pred != y_truth, 0], x[y_pred != y_truth, 1], "ko", markersize=12, fillstyle="none")

plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors="k")

plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors=["g", "r"],
             alpha=0.3)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors=["b", "r"],
             alpha=0.3)
plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors=["b", "g"],
             alpha=0.3)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

