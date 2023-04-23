import numpy as np
import pandas as pd
import streamlit as st

st.write("# Welcome to the Insurance Cross-Selling Prediction Web App! 👋")

st.markdown("""
        **👈 Select a page from the dropdown on the left** to see some applications of the app!"""
        )

st.markdown("""
Insurance Companies are becoming data-driven oriented with the Marketing field assuming a strategic role for the Company growth.
The project starts with the goal to predict whether health insurance policyholders are interested to buy a vehicle insurance coverage. 
The task goes ahead profiling customers.
There are many ways to generate additional revenue for a Company: introducing new products, offering additional services, or even raising prices. 
One technique known as cross-selling can lead to increase customer lifetime value.
Cross-selling is a marketing strategy used to get a buyer to spend more by purchasing a product that’s related and/or supplementary 
to what’s being bought already. 
In the first step, the cross-selling prediction activity, has been used calibrated classifiers with Logistic Regression 
employed as a benchmark model and it has been compared with other machine learning models such as Naïve Bayes and Hist 
Gradient Boosting Machine. The ensemble model shows the best performance. In the second step, 
has been profiled customers interested in the purchasing coverage using K-means clustering method on numerical features 
and then applying the split to the all data set. In this way has been possible to understand relationships between 
numerical features as Annaul Premium and Age with categorical features as Vehicle Age, Gender and Vehicle Damage. 
Machine Learning tools are relevant because they give the opportunity to allocate in a good way marketing budget resources, 
and reducing wastes. The introduction of modern machine learning models are required to improve the overall marketing 
strategy overcoming the poor performance.
Actuaries play a relevant role as a joint ring between Actuarial department and Marketing department providing their 
expert judgment in the prediction and in the evaluation of the risk given some features involved are also applied 
in the Risk Premium calculation of traditional models and to exploit the use of big data to satisfy customers’ needs.
""")
