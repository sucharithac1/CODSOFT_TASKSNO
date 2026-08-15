import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.style.use("seaborn")
sns.set_palette("Set2")


os.makedirs("output", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


df = pd.read_csv("cleaned_data.csv")  
print("Dataset Loaded Successfully!\n")
print(df.head())


print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include="all"))


print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())


df["Sales"] = df["Quantity"] * df["Price"]
print("\nSales column added successfully!")
print(df.head())

# Step 5: Explore Distributions
df[["Quantity","Price","Sales"]].hist(bins=30, figsize=(15,10))
plt.suptitle("Feature Distributions")
plt.savefig("visualizations/distributions.png")
plt.close()


for col in ["Product","Category","City","Payment_Method"]:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head())


corr_matrix = df[["Quantity","Price","Sales"]].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap.png")
plt.close()


sns.scatterplot(x="Quantity", y="Price", data=df)
plt.title("Quantity vs Price")
plt.savefig("visualizations/scatter_quantity_price.png")
plt.close()


sns.scatterplot(x="Quantity", y="Sales", data=df)
plt.title("Quantity vs Sales")
plt.savefig("visualizations/scatter_quantity_sales.png")
plt.close()


sns.boxplot(x=df["Price"])
plt.title("Boxplot of Price")
plt.savefig("visualizations/boxplot_price.png")
plt.close()

sns.boxplot(x=df["Sales"])
plt.title("Boxplot of Sales")
plt.savefig("visualizations/boxplot_sales.png")
plt.close()


sales_by_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
print("\nSales by Category:")
print(sales_by_category)

sales_by_city = df.groupby("City")["Sales"].sum().sort_values(ascending=False)
print("\nSales by City:")
print(sales_by_city)

avg_sales_by_product = df.groupby("Product")["Sales"].mean().sort_values(ascending=False)
print("\nAverage Sales by Product:")
print(avg_sales_by_product)


df.describe(include="all").to_csv("output/eda_summary.csv")
print("\nEDA Summary Report Saved as output/eda_summary.csv")
print("All plots saved in visualizations/ folder.")
