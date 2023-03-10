#!/usr/bin/env python
# coding: utf-8

# In[49]:


import streamlit as st
import pickle as pkl
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


# In[35]:


lrs=pkl.load(open('linear_model.pkl','rb'))
df=pkl.load(open('mart_df.pkl','rb'))


# In[8]:


st.title("Amazing Mart Profit Prediction")


# In[43]:


# Discount
Discount=st.selectbox('Discount',df['Discount'])
# Actula Discount
Actual_Discount=st.selectbox('Actual Discount',df['Actual Discount'])
# Product Name
Product_Name=st.slider('select Product Name',0,1615)
# Sales
Sales=st.slider('select Sales',0,3759)
# Category
Category=st.selectbox('Category',df['Category'].unique())
# Sub-Category
Sub_Category=st.selectbox('Sub-Category',df['Sub-Category'].unique())
# City
City=st.slider('select City',0,999)
# Country
Country=st.selectbox('Country',df['Country'].unique())
# Region
Region=st.selectbox('Region',df['Region'].unique())
#Segment
Segment=st.selectbox('Segment',df['Segment'].unique())
#State
State=st.slider('select State',0,126)
# Quantity
Quantity=st.selectbox('Quantity',df['Quantity'])
# Order ID
Order_ID=st.slider('Order ID',0,3759)


# In[46]:


query = np.array([Discount,Actual_Discount,Product_Name,Sales,Category,Sub_Category,City,Country,Region,Segment,State,Quantity,Order_ID])
query = query.reshape(1,13)
st.title("The predict price of this configuration is "+ str(int(lrs.predict(query)[0])))

