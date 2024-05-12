#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(student_grades, bins=bins, edgecolor="black")
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xticks(bins)
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.show()