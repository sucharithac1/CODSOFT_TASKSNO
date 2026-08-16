import pandas as pd
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


VISUALIZATION_DIR = os.path.join(
    BASE_DIR,
    "visualizations"
)


os.makedirs(VISUALIZATION_DIR, exist_ok=True)




df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")
print("\nFirst 5 rows:")
print(df.head())




df["Sales"] = df["Quantity"] * df["Price"]

print("\nSales column added successfully!")




sales_by_category = (
    df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x="Category",
    y="Sales",
    data=sales_by_category
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "sales_by_category.png"
    )
)

plt.close()



sales_by_city = (
    df.groupby("City")["Sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="City",
    y="Sales",
    data=sales_by_city
)

plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "sales_by_city.png"
    )
)

plt.close()




payment_counts = df["Payment_Method"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    payment_counts,
    labels=payment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Payment Method Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "payment_methods.png"
    )
)

plt.close()




df["Order_Date"] = pd.to_datetime(
    df["Order_Date"]
)

df["Month"] = (
    df["Order_Date"]
    .dt.to_period("M")
)

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)


monthly_sales["Month"] = (
    monthly_sales["Month"]
    .astype(str)
)

plt.figure(figsize=(10, 6))

sns.lineplot(
    x="Month",
    y="Sales",
    data=monthly_sales,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATION_DIR,
        "monthly_sales_trend.png"
    )
)

plt.close()




print("\nTask 3 Visualizations generated successfully!")

print("\nFiles saved in:")
print(VISUALIZATION_DIR)

print("\nGenerated files:")
print("1. sales_by_category.png")
print("2. sales_by_city.png")
print("3. payment_methods.png")
print("4. monthly_sales_trend.png")
