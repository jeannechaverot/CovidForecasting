# Forecasting  the  Covid-19  outbreak  using  ConstrainedQuadratic  Optimization

This project has been done from February to June of 2020, in the scope of my Semester project at EPFL (Ecole Polytechniques Fédérale de Lausanne) under the supervision of Pf. Guillaume Obozinski and Pr. Olivier Verscheure from the Swiss Data Science Center. (12 CTS).

## Data

For up to date data, go on the DATA/COVID-19 folder and git pull. It will update the data-sets on the number of confirmed, deaths, and recovered cases.


### 0. Data Cleaning

In this notebook, we explore the data. We also choose:
- which country we are working on
- which time interval will we use in our model in order to make our predictions

Data cleaning is performed to correct data inconsistencies.

### 1.Lasso, Ridge, Elastic Net Regularizations

Perform basic linear regression, without and with regularization.

### 2. 2.2 [DEATHS] First Formulation

Find best smoothing parameters. Use first formulation of quadratic optimization as detailed in paper to make predictions. 

### 2. 2.2 [DEATHS] Second Formulation

Find best smoothing parameters. Use second formulation of quadratic optimization as detailed in paper to make predictions. 
