# Air Pollution Data Analysis and Prediction

## Problem Statement
Air pollution levels fluctuate due to multiple environmental factors. Raw data alone is not useful unless processed and analyzed. This project focuses on extracting insights and building a basic prediction model from air quality data.

---

## What This Project Does
- Loads and cleans air pollution dataset
- Performs exploratory data analysis (EDA)
- Generates visual insights (saved as figures)
- Builds a Linear Regression model
- Evaluates prediction performance using R² score

---

## Dataset
- File: `Air Pollution.csv`
- Contains pollution-related attributes (e.g., PM levels, gases, etc.)
- Used as input for analysis and model training

---

## Tech Stack
- Python
- Pandas (data handling)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning)

---

## Workflow

1. Data Loading
Dataset is loaded using Pandas:
python
df = pd.read_csv("Air Pollution.csv")
2. Data Cleaning
Checked for missing values
Removed duplicates
Verified column data types
3. Exploratory Data Analysis
Distribution plots
Correlation heatmap
Outlier detection
4. Visualization Output

Saved figures:

Figure_1.png to Figure_9.png

These represent trends, distributions, and relationships in data.

5. Model Building
Applied Linear Regression
Split dataset into training and testing sets
6. Evaluation
Used R² Score to measure performance

