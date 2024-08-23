import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
sales = pd.read_csv('sales.csv')
product_group = pd.read_csv('product_group.csv')
website_access = pd.read_csv('website_access.csv')

# Convert SaleDate to datetime format
sales['SaleDate'] = pd.to_datetime(sales['SaleDate'])

# 1. Line Chart: Sales Over Time
sales_by_date = sales.groupby(sales['SaleDate'].dt.date)['Quantity'].sum()

plt.figure(figsize=(10, 6))
plt.plot(sales_by_date.index, sales_by_date.values, marker='o', linestyle='-', color='b')
plt.title('Total Products Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar Chart: Revenue by Product Group
sales_product_group = sales.merge(product_group, left_on='ProductDetailID', right_on='ProductGroupID')
sales_by_group = sales_product_group.groupby('GroupName')['TotalAmount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sales_by_group.plot(kind='bar', color='orange')
plt.title('Revenue by Product Group')
plt.xlabel('Product Group')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Pie Chart: Website Access by User Type
access_by_type = website_access['AccessType'].value_counts()

plt.figure(figsize=(8, 8))
access_by_type.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'salmon'])
plt.title('Website Access Distribution by User Type')
plt.ylabel('')
plt.tight_layout()
plt.show()
