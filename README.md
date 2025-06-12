# Basic Data Analysis with Python

## 📊 Project Overview

This repository showcases fundamental **data analysis** skills using **Python**, focusing on common techniques to explore, visualize, and model data. It demonstrates a practical approach to understanding datasets, uncovering insights, and building predictive models.

The primary goal of this project is to illustrate:

* **Data loading and initial exploration** with `pandas`.
* **Descriptive statistics and data summarization**.
* **Data visualization** to identify patterns and distributions using `matplotlib`.
* **Feature selection** for predictive modeling.
* **Basic machine learning application** (Logistic Regression) for classification.
* **Model evaluation** using metrics like accuracy and confusion matrices.

## ✨ Skills Demonstrated

This project highlights proficiency in:

* **Python Programming**: Core language proficiency for data manipulation and analysis.
* **Data Science Fundamentals**: Understanding of data exploration, feature engineering, and model building.
* **Pandas**: Efficient data handling, cleaning, and transformation.
* **NumPy**: Numerical operations and array manipulation.
* **Matplotlib**: Creating various plots for data visualization (histograms, scatter plots, box plots).
* **Scikit-learn**: Implementing machine learning models (Logistic Regression) and evaluating performance.
* **Statistical Analysis**: Grouping data, calculating means, and understanding distributions.

## 🚀 How to Run This Project

To explore this analysis on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thexqin/basic-data-analysis.git
    cd basic-data-analysis
    ```
    
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas openpyxl numpy scikit-learn matplotlib
    ```

4.  **Open and run the analysis:**
    You can open the Jupyter Notebook to step through the analysis.
    *(You will need to ensure the .xlsx file is in the same directory as your script.)*

## 📈 Analysis Walkthrough & Key Findings

This project uses a dataset to analyze factors related to student placements and salaries.

### Data Loading and Initial Inspection

The analysis begins by loading the dataset into a Pandas DataFrame and performing an initial inspection using `df.info()`. This step is crucial for understanding data types, non-null counts, and memory usage.

```python
import pandas as pd
import numpy as np

df = pd.read_excel('dean.xlsx')
print(df.info())
```

### Exploratory Data Analysis (EDA)

Histograms are generated for all numerical columns to visualize their distributions, providing quick insights into the spread and patterns of the data.

```python
import matplotlib.pyplot as plt
df.hist(figsize=(12,12))
plt.tight_layout() # Adjust layout to prevent overlapping titles
plt.show() # Display the plot
```

Further EDA delves into the relationship between placement status (`Placement_B`) and `Salary`.

```python
print(df.groupby('Placement_B')['Salary'].mean())
print(df.groupby('Placement_B')['Salary'].count())
```
**Key Insight:** The analysis clearly shows a significant difference in salary based on placement status, with placed individuals having a mean salary of $274,550.

### Deep Dive into Placed Candidates' Salaries

A subset of the data focusing only on placed candidates (`Placement_B == 1`) is created to analyze salary distribution more closely.

```python
df_placed = df[df['Placement_B']==1]
df_placed['Salary'].hist(figsize=(8,4), bins=35)
plt.title('Salary Distribution for Placed Candidates')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show() # Display the plot

df_placed['Salary'].plot.box(figsize=(8,2), vert=False)
plt.title('Box Plot of Salary for Placed Candidates')
plt.xlabel('Salary')
plt.show() # Display the plot
```

Scatter plots are used to visually inspect potential relationships between various academic performance metrics (e.g., `Percent_SSC`, `Percent_HSC`, `Percent_Degree`, `Percentile_ET`, `Percent_MBA`) and `Salary` for placed individuals.

```python
plt.figure(figsize=(8,8))
plt.scatter(df_placed['Percent_SSC'], df_placed['Salary'], label='SSC %')
plt.scatter(df_placed['Percent_HSC'], df_placed['Salary'], label='HSC %')
plt.scatter(df_placed['Percent_Degree'], df_placed['Salary'], label='Degree %')
plt.scatter(df_placed['Percentile_ET'], df_placed['Salary'], label='ET Percentile')
plt.scatter(df_placed['Percent_MBA'], df_placed['Salary'], label='MBA %')
plt.title('Academic Performance vs. Salary for Placed Candidates')
plt.xlabel('Percentage/Percentile')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

### Predictive Modeling (Logistic Regression)

A **Logistic Regression** model is implemented to predict `Placement_B` (placement status) based on a selection of academic and communication-related features.

**Features (X):** `Percent_SSC`, `Percent_HSC`, `Percent_Degree`, `Percentile_ET`, `Percent_MBA`, `Marks_Communication`, `Marks_Projectwork`, `Marks_BOCA`
**Target (y):** `Placement_B`

```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X = df[['Percent_SSC', 'Percent_HSC', 'Percent_Degree', 'Percentile_ET',
        'Percent_MBA', 'Marks_Communication', 'Marks_Projectwork', 'Marks_BOCA']]
y = df['Placement_B']

lr = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
lr.fit(X, y)

print(f"Model Accuracy: {lr.score(X, y):.3f}")

predict_test = lr.predict(X)
cmtx = pd.DataFrame(
    metrics.confusion_matrix(y, predict_test),
    index=['actual:not_placed', 'actual:placed'],
    columns=['pred:not_placed', 'pred:placed'])
print("\nConfusion Matrix:")
print(cmtx)
```

**Model Accuracy & Confusion Matrix:** The model achieved an accuracy of approximately **79.5%**. The confusion matrix provides a breakdown of correct and incorrect predictions for placement status.

## 📂 Repository Structure

```
basic-data-analysis/
├── a_dean's_dilemma_selection_of_students_for_the_MBA_program.xlsx   # Sample dataset used for analysis
├── a_dean's_dilemma_analysis.ipynb                                   # Jupyter Notebook with the full analysis
└── README.md                                                         # This file
```

## 🤝 Contribution

Feel free to fork this repository, experiment with the code, and suggest improvements or additional analyses!
