import pandas as pd 
import matplotlib.pyplot as plt
df=pd.read_csv("sales_data.csv")
print(df.head())

print("Missing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)
# Fill missing values
df.fillna({
    "Quantity": 0,
    "Price": 0,
    "Category": "Unknown"
}, inplace=True)

# Calculate Revenue
df["Revenue"] = df["Quantity"] * df["Price"]
print("\nRevenue Column Added:")
print(df[["CustomerName", "Product", "Revenue"]].head())

top_customers = (
    df.groupby("CustomerName")["Revenue"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Customers:")
print(top_customers)

# chart for revenue
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["Revenue"].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.savefig("monthly_sales.png")

# Graph for top 10 customers
plt.figure(figsize=(10,5))
top_customers.plot(kind="bar")
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer")
plt.ylabel("Revenue")
plt.savefig("top_customers.png")

plt.show()

