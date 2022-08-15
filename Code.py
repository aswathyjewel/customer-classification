#!/usr/bin/env python
# coding: utf-8

# In[81]:


#---------------------------------------------------------------
# @author Bhargav Vemula C0818081
# @author Govind Vijayan C0819805
# @author Jaison Kayamkattil Jacob C0814631
# @author Prasanth Moothedath Padmakumar C0796752

# Dataset File downloaded from https://archive.ics.uci.edu/ml/datasets/online+retail

# Description of each attributes are as follows
# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
# Description: Product (item) name. Nominal.
# Quantity: The quantities of each product (item) per transaction. Numeric.
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
# UnitPrice: Unit price. Numeric, Product price per unit in sterling.
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
# Country: Country name. Nominal, the name of the country where each customer resides.
#---------------------------------------------------------------


# ### Import Libraries

# In[1]:


# Pandas library for data manipulation and analysis
# Numpy library for some standard mathematical functions
# Matplotlib library to visualize the data in the form of different plot
# Seaborn library for visualizing statistical graphics and work on top of Matplotlib
# Datetime library for date manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# ### Read the dataset from csv file

# In[4]:


df = pd.read_excel('Online Retail.xlsx')


# In[5]:


# Display first 5 rows of the dataset using head function
df.head()


# In[6]:


df.shape


# ### Checking data set for null values and duplicate entries

# In[7]:


# Print summary of the dataframe 
df.info()


# In[8]:


#Check for missing values in the dataset (axis=0 implies that the sum operation is performed on the columns)
df.isnull().sum(axis=0)


# In[9]:


# Removing entries with customer  as null
df = df[df['CustomerID'].notna()]


# In[10]:


# Checking sum of duplicated entries
df.duplicated().sum()


# In[11]:


# Removing all duplicate entries
df.drop_duplicates(keep=False,inplace=True)
df.shape


# In[12]:


#Validate if there are any negative values in Quantity column
df.Quantity.min()


# In[13]:


#Filter out records with negative values
df = df[(df['Quantity']>0)]


# In[14]:


#Validate if there are any negative values in UnitPrice column
df.UnitPrice.min()


# In[15]:


df.shape


# ### Data Preprocessing
# #### Create RFM Table

# In[16]:


# Create TotalSum colummn
df["TotalSum"] = df["Quantity"] * df["UnitPrice"]

# Create date variable that records recency
snapshot_date = max(df.InvoiceDate) + datetime.timedelta(days=1)

# Aggregate data by each customer
customers = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename columns
customers.rename(columns = {'InvoiceDate': 'Recency',
                            'InvoiceNo': 'Frequency',
                            'TotalSum': 'MonetaryValue'}, inplace=True)


# In[17]:


customers.head()


# #### Manage Skewness

# In[18]:


# Checking th skewness of data
fig, ax = plt.subplots(1, 3, figsize=(15,3))
sns.histplot(customers['Recency'], ax=ax[0])
sns.histplot(customers['Frequency'], ax=ax[1])
sns.histplot(customers['MonetaryValue'], ax=ax[2])
plt.tight_layout()
plt.show()


# In[19]:


# Trying out log transformation, square root transformation, and box-cox transformation for recency and frequency
from scipy import stats
def analyze_skewness(x):
    fig, ax = plt.subplots(2, 2, figsize=(5,5))
    sns.histplot(customers[x], ax=ax[0,0])
    sns.histplot(np.log(customers[x]), ax=ax[0,1])
    sns.histplot(np.sqrt(customers[x]), ax=ax[1,0])
    sns.histplot(stats.boxcox(customers[x])[0], ax=ax[1,1])
    plt.tight_layout()
    plt.show()
    
    print(customers[x].skew().round(2))
    print(np.log(customers[x]).skew().round(2))
    print(np.sqrt(customers[x]).skew().round(2))
    print(pd.Series(stats.boxcox(customers[x])[0]).skew().round(2))


# In[20]:


analyze_skewness('Recency')


# In[21]:


analyze_skewness('Frequency')


# In[22]:


# Trying cubic root transformation on monetary value
fig, ax = plt.subplots(1, 2, figsize=(10,3))
sns.histplot(customers['MonetaryValue'], ax=ax[0])
sns.histplot(np.cbrt(customers['MonetaryValue']), ax=ax[1])
plt.show()
print(customers['MonetaryValue'].skew().round(2))
print(np.cbrt(customers['MonetaryValue']).skew().round(2))


# In[23]:


pd.Series(np.cbrt(customers['MonetaryValue'])).values


# In[24]:


# Set the Numbers
customers_fix = pd.DataFrame()
customers_fix["Recency"] = stats.boxcox(customers['Recency'])[0]
customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
customers_fix.tail()


# #### Centering and Scaling Variables

# In[25]:


# Scaling th data for normalising
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)
print(customers_normalized.mean(axis = 0).round(2))
print(customers_normalized.std(axis = 0).round(2))


# In[26]:


pd.DataFrame(customers_normalized).head()


# ### Modelling
# #### Choose k-number

# In[27]:


# Finding the K number using elbow method
from sklearn.cluster import KMeans

sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[28]:


# Chose k as 3
model = KMeans(n_clusters=3, random_state=42)
model.fit(customers_normalized)


# In[29]:


customers.shape


# ### Cluster Analysis

# In[30]:


# add cluster column to customer data frame
customers["Cluster"] = model.labels_
customers.head()


# In[31]:


# Grouing customers data frame based on their clusters and mean of resoective R, F and M
customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(1)


# ### Snake Plots

# In[32]:


# Using the normalised value for plotting the snake plot
df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_
df_normalized.head()


# In[39]:


# Melting The Data to plot it in snake plot
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()


# In[34]:


# snake plot
sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)


# ### ANALYSIS
# #### Cluster 0 : Churning customers 
# #### Cluster 1 : Actively Loyal customers
# #### Cluster 2 : Potential Loyal customers
