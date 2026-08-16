import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
sns.set_palette("Set2")


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_sales_data.csv"
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

VISUALIZATION_DIR = os.path.join(
    BASE_DIR,
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("==========================================")
print("       DATASET LOADED SUCCESSFULLY")
print("==========================================")

print("\nFirst 5 Rows:")
print(df.head())

print("\n==========================================")
print("           DATASET INFORMATION")
print("==========================================")

print("\nShape of Dataset:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nInfo:")
df.info()

print("\n==========================================")
print("          SUMMARY STATISTICS")
print("==========================================")

print(df.describe(include="all"))


print("\n==========================================")
print("             MISSING VALUES")
print("==========================================")

missing_values = df.isnull().sum()

print(missing_values)

print("\n==========================================")
print("             DUPLICATE ROWS")
print("==========================================")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


df["Sales"] = df["Quantity"] * df["Price"]

print("\n==========================================")
print("          SALES COLUMN CREATED")
print("==========================================")

print(df[["Quantity", "Price", "Sales"]].head())

print("\nCreating Feature Distribution plots...")

df[
    ["Quantity", "Price", "Sales"]
].hist(
    bins=30,
    figsize=(15, 10)
)

plt.suptitle("Feature Distributions")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "distributions.png"
    )
)

plt.close()




print("\n==========================================")
print("             VALUE COUNTS")
print("==========================================")

columns_to_check = [
    "Product",
    "Category",
    "City",
    "Payment_Method"
]

for col in columns_to_check:

    if col in df.columns:

        print(f"\nValue counts for {col}:")

        print(
            df[col]
            .value_counts()
            .head()
        )



print("\n==========================================")
print("          CORRELATION MATRIX")
print("==========================================")

corr_matrix = df[
    ["Quantity", "Price", "Sales"]
].corr()

print(corr_matrix)




plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "heatmap.png"
    )
)

plt.close()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="Quantity",
    y="Price",
    data=df
)

plt.title("Quantity vs Price")
plt.xlabel("Quantity")
plt.ylabel("Price")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "scatter_quantity_price.png"
    )
)

plt.close()

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="Quantity",
    y="Sales",
    data=df
)

plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "scatter_quantity_sales.png"
    )
)

plt.close()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Price"]
)

plt.title("Boxplot of Price")
plt.xlabel("Price")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "boxplot_price.png"
    )
)

plt.close()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Sales"]
)

plt.title("Boxplot of Sales")
plt.xlabel("Sales")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "boxplot_sales.png"
    )
)

plt.close()


sales_by_category = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n==========================================")
print("           SALES BY CATEGORY")
print("==========================================")

print(sales_by_category)


sales_by_city = (
    df.groupby("City")["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n==========================================")
print("              SALES BY CITY")
print("==========================================")

print(sales_by_city)

avg_sales_by_product = (
    df.groupby("Product")["Sales"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print("\n==========================================")
print("        AVERAGE SALES BY PRODUCT")
print("==========================================")

print(avg_sales_by_product)

sales_by_category.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sales_by_category.csv"
    )
)




sales_by_city.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sales_by_city.csv"
    )
)




avg_sales_by_product.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "average_sales_by_product.csv"
    )
)





summary = df.describe(
    include="all"
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eda_summary.csv"
    )
)



missing_values.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_values.csv"
    )
)




print("\n==========================================")
print("          TASK 2 EDA COMPLETED")
print("==========================================")

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nVisualizations saved in:")
print(VISUALIZATION_DIR)

print("\nGenerated visualizations:")

print("1. distributions.png")
print("2. heatmap.png")
print("3. scatter_quantity_price.png")
print("4. scatter_quantity_sales.png")
print("5. boxplot_price.png")
print("6. boxplot_sales.png")

print("\nGenerated output files:")

print("1. eda_summary.csv")
print("2. missing_values.csv")
print("3. sales_by_category.csv")
print("4. sales_by_city.csv")
print("5. average_sales_by_product.csv")

print("\nTask 2 completed successfully!")
