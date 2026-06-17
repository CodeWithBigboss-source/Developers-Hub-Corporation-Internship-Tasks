# Task 1: Exploring and Visualizing a Simple Dataset

## Objective

The objective of this task is to learn how to load, inspect, analyze, and visualize a dataset to understand data trends and distributions.

## Dataset Used

* Iris Dataset (loaded using Seaborn)

## Technologies Used

* Python
* Pandas
* Seaborn
* Matplotlib

## Steps Performed

### 1. Loaded the Dataset

* Loaded the Iris dataset using `sns.load_dataset()`.

### 2. Inspected the Dataset

* Displayed dataset shape.
* Displayed column names.
* Displayed the first five rows.

### 3. Generated Summary Statistics

* Used `df.info()` to inspect data types and missing values.
* Used `df.describe()` to generate statistical summaries.

### 4. Created Visualizations

#### Scatter Plot

* Visualized the relationship between `sepal_length` and `sepal_width`.
* Colored points according to species.

#### Histograms

* Displayed value distributions for all numerical features.

#### Box Plot

* Identified potential outliers in numerical features.

## Files

* `iris_visualization.py` → Python code implementation.
* `README.md` → Task documentation.

## Learning Outcomes

* Data loading using Pandas and Seaborn
* Basic Exploratory Data Analysis (EDA)
* Descriptive statistics
* Data visualization using Matplotlib and Seaborn
