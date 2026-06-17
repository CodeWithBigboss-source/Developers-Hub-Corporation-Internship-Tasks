# ==========================
# Import Libraries
# ==========================

# Import pandas for working with data tables
import pandas as pd

# Import seaborn for visualization and built-in datasets
import seaborn as sns

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt


# ==========================
# Load Dataset
# ==========================

# Load the Iris dataset from seaborn
df = sns.load_dataset("iris")

# Print confirmation message
print("Iris dataset loaded successfully")


# ==========================
# Dataset Inspection
# ==========================

# Print number of rows and columns
print("\nShape:", df.shape)

# Print column names
print("\nColumns:")
print(df.columns)

# Print first 5 rows
print("\nFirst 5 rows:")
print(df.head())


# ==========================
# Dataset Summary Statistics
# ==========================

# Display dataset information
print("\nDataset Information:")
df.info()

# Display statistical summary
print("\nSummary Statistics:")
print(df.describe())


# ==========================
# Scatter Plot
# ==========================

# Create a figure
plt.figure(figsize=(8,5))

# Create scatter plot
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue="species"
)

# Add title
plt.title("Sepal Length vs Sepal Width")

# Add x-axis label
plt.xlabel("Sepal Length")

# Add y-axis label
plt.ylabel("Sepal Width")

# Display graph
plt.show()


# ==========================
# Histograms
# ==========================

# Create histograms for numerical columns
df.hist(figsize=(10,8))

# Adjust spacing
plt.tight_layout()

# Display graphs
plt.show()


# ==========================
# Box Plot
# ==========================

# Create figure
plt.figure(figsize=(10,6))

# Create box plot for numerical columns
sns.boxplot(data=df.select_dtypes(include="number"))

# Add title
plt.title("Box Plot of Iris Features")

# Display graph
plt.show()