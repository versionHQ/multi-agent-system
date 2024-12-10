import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)

warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(ROOT_DIR, "/sample_dataset/Online Sales Data.csv"))

df.isna().sum()
df.duplicated().sum()
df.drop(columns="Transaction ID", inplace=True)
df.info()
df.Date.min()
df.Date.max()
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["day_name"] = df["Date"].dt.day_name()
df["Product Category"].unique()
df["Product Category"].value_counts()
df["Product Name"].nunique()
df[df["Product Category"] == "Electronics"]["Product Name"]
df[df["Product Category"] == "Home Appliances"]["Product Name"]
df[df["Product Category"] == "Clothing"]["Product Name"]
df[df["Product Category"] == "Books"]["Product Name"]
df[df["Product Category"] == "Beauty Products"]["Product Name"]
df[df["Product Category"] == "Sports"]["Product Name"]
df["Payment Method"].value_counts()
colors = plt.get_cmap("Pastel1").colors
df["Payment Method"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", colors=colors, explode=(0.05, 0, 0)
)
df.Region.unique()
df.Region.value_counts()
colors = plt.get_cmap("Pastel1_r").colors
df["Region"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", colors=colors, explode=(0, 0, 0), startangle=90
)
sales_by_region = df.groupby("Region")["Units Sold"].sum().reset_index()
print(sales_by_region)

sns.barplot(data=sales_by_region, x="Region", y="Units Sold")

total_sales_by_category = (
    df.groupby("Product Category")["Units Sold"].sum().reset_index()
)
print(total_sales_by_category)
plt.figure(figsize=(10, 5))
sns.barplot(data=total_sales_by_category, x="Product Category", y="Units Sold")

total_sales_Payment_Method = (
    df.groupby("Payment Method")["Units Sold"].sum().reset_index()
)
plt.figure(figsize=(7, 5))
sns.barplot(data=total_sales_Payment_Method, x="Payment Method", y="Units Sold")

df["Units Sold"].max()
df[df["Units Sold"] == df["Units Sold"].max()]
data = df[df["Product Category"] == "Electronics"]

Product_Units = (
    data.groupby("Product Name")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

data[data["Units Sold"].isin(Product_Units["Units Sold"].nlargest(8))]
data = df[df["Product Category"] == "Clothing"]
Product_Units = (
    data.groupby("Product Name")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
data[data["Units Sold"].isin(Product_Units["Units Sold"].nlargest(3))].sort_values(
    by="Units Sold", ascending=False
)
plt.figure(figsize=(10, 5))
data.groupby("day_name")["Units Sold"].sum()
units_sold_by_day = data.groupby("day_name")["Units Sold"].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=units_sold_by_day, x="day_name", y="Units Sold")

plt.title("Total Units Sold by Day of the Week of Clothing")
plt.xlabel("Day of the Week")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

region_counts = data.Region.value_counts()

for region, count in region_counts.items():
    percentage = (count / region_counts.sum()) * 100
    print(f"{region}: {percentage:.1f}%")
Payment_Method = data["Payment Method"].value_counts()

for region, count in Payment_Method.items():
    percentage = (count / Payment_Method.sum()) * 100
    print(f"{region}: {percentage:.1f}%")

data = df[df["Product Category"] == "Books"]
plt.figure(figsize=(10, 5))
data.groupby("day_name")["Units Sold"].count().sort_values(ascending=False)
units_sold_by_day = data.groupby("day_name")["Units Sold"].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=units_sold_by_day, x="day_name", y="Units Sold")

plt.title("Total Units Sold by Day of the Week of Books")
plt.xlabel("Day of the Week")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


region_counts = data.Region.value_counts()

for region, count in region_counts.items():
    percentage = (count / region_counts.sum()) * 100
    print(f"{region}: {percentage:.1f}%")

Payment_Method = data["Payment Method"].value_counts()

for region, count in Payment_Method.items():
    percentage = (count / Payment_Method.sum()) * 100
    print(f"{region}: {percentage:.1f}%")

df_grouped = df.groupby(["month"])["Units Sold"].sum().reset_index()
df_grouped.set_index(["month"], inplace=True)
df_grouped.plot(kind="bar", color=["#FFD7C4"])

df_grouped = df.groupby(["day_name"])["Units Sold"].sum().reset_index()
df_grouped.set_index(["day_name"], inplace=True)
df_grouped.plot(kind="bar", color="#F1DEC6")

df_grouped = (
    df.groupby(["day_name", "Payment Method"])["Units Sold"].sum().reset_index()
)
df_grouped.set_index(["day_name", "Payment Method"], inplace=True)
df_grouped = (
    df.groupby(["day_name", "Payment Method"])["Units Sold"].sum().reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_grouped,
    x="day_name",
    y="Units Sold",
    hue="Payment Method",
    palette=["#F1DEC6", "#FFD7C4", "#FFDBB5"],
    edgecolor="black",
)

plt.title("Units Sold by Payment Method and Day of the Week")
plt.xlabel(" day name")
plt.ylabel("Units Sold")
plt.legend(title="Payment Method")
plt.tight_layout()
plt.show()

df_grouped = df.groupby(["month", "Payment Method"])["Units Sold"].sum().reset_index()
df_grouped.set_index(["month", "Payment Method"], inplace=True)
df_grouped = df.groupby(["month", "Payment Method"])["Units Sold"].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_grouped,
    x="month",
    y="Units Sold",
    hue="Payment Method",
    palette=["#F1DEC6", "#FFD7C4", "#FFDBB5"],
    edgecolor="black",
)

plt.title("Units Sold by Payment Method and month")
plt.xlabel("month")
plt.ylabel("Units Sold")
plt.legend(title="Payment Method")
plt.tight_layout()
plt.show()

data["Unit Price"].describe()
data["Total Revenue"].describe()

plt.figure(figsize=(5, 4))
sns.scatterplot(x="Unit Price", y="Total Revenue", data=df)

plt.title("Scatter Plot of Total Revenue vs Unit Price")
plt.xlabel("Unit Price")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.show()

df.groupby(["Product Category"])["Total Revenue"].sum().reset_index()
df_grouped = (
    df.groupby(["Product Category", "Region"])["Total Revenue"].sum().reset_index()
)
df_grouped = (
    df.groupby(["Region", "Product Category"])["Total Revenue"].sum().reset_index()
)
df_grouped.set_index(["Region", "Product Category"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "Product Category"])["Total Revenue"]
    .sum()
    .reset_index()
)
df_grouped.set_index(["Payment Method", "Product Category"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "Region"])["Total Revenue"].sum().reset_index()
)
df_grouped.set_index(["Payment Method", "Region"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "Region", "Product Category"])["Units Sold"]
    .sum()
    .reset_index()
)
df_grouped.set_index(["Payment Method", "Region"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "Region", "Product Category"])["Total Revenue"]
    .sum()
    .reset_index()
)
df_grouped.set_index(["Payment Method", "Region", "Product Category"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "day_name"])["Total Revenue"].sum().reset_index()
)
df_grouped.set_index(["Payment Method", "day_name"], inplace=True)
df_grouped = (
    df.groupby(["Payment Method", "month"])["Total Revenue"].sum().reset_index()
)
df_grouped.set_index(["Payment Method", "month"], inplace=True)
df["Product Category"].unique()

revenue_summary = (
    df.groupby("Product Category")["Total Revenue"].agg(["min", "max"]).reset_index()
)
revenue_summary = (
    df.groupby(["Payment Method", "Product Category"])["Total Revenue"]
    .agg(["min", "max"])
    .reset_index()
)
revenue_summary.set_index(["Payment Method", "Product Category"], inplace=True)
print(revenue_summary)
