<div align="center">

# 🛍️ Customer Segmentation & Behavioral Analysis

### *Marketing Campaign Dataset – Exploratory Data Analysis & Clustering*

**By Gheffari Nour El Houda**

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br>

![Project Status](https://img.shields.io/badge/status-completed-success?style=flat-square)
![Type](https://img.shields.io/badge/type-unsupervised%20learning-blueviolet?style=flat-square)
![Domain](https://img.shields.io/badge/domain-marketing%20analytics-ff69b4?style=flat-square)

</div>

---

> 🎯 **TL;DR** — This project segments retail customers using unsupervised machine learning (K-Means + PCA + t-SNE) on a marketing campaign dataset of 2,240 customers, identifying three actionable clusters: **VIPs**, **Mid-Value Growth**, and **Price-Sensitive** customers — each paired with a targeted marketing strategy.

---

## 📑 Table of Contents

**Introduction**

- [🏪 Business Context](#business-context)
- [❓ Main Questions We Aim to Answer](#main-questions-we-aim-to-answer)
- [🎯 Project Objective](#project-objective)
- [🧭 Methodology Overview](#methodology-overview)

**Analysis Pipeline**

- [📥 1. Data Loading & Environment Setup](#1-data-loading-environment-setup)
- [⚡ 2. Memory Optimization](#2-memory-optimization)
- [🔍 3. Initial Data Exploration](#3-initial-data-exploration)
- [🧹 4. Data Cleaning](#4-data-cleaning)
- [🛠️ 5. Feature Engineering](#5-feature-engineering)
- [🎯 6. Outlier Detection & Removal](#6-outlier-detection-removal)
- [🎯 7. Outlier Detection & Removal](#7-outlier-detection-removal)
- [📊 8. Univariate Analysis](#8-univariate-analysis)
- [📈 8. Bivariate Analysis](#8-bivariate-analysis)
- [🔢 9. Feature Encoding](#9-feature-encoding)
- [🌐 10. Multivariate Analysis](#10-multivariate-analysis)
- [📐 11. Scaling & Dimensionality Reduction](#11-scaling-dimensionality-reduction)
- [🗺️ 12. t-SNE Visualization](#12-t-sne-visualization)
- [🎨 13. Customer Segmentation Using K-Means](#13-customer-segmentation-using-k-means)
- [👥 14. Cluster Profiling & Interpretation](#14-cluster-profiling-interpretation)
- [💼 15. Customer Segmentation – Business Strategies & Actions](#15-customer-segmentation-business-strategies-actions)

---

## 🏪 Business Context

Customer segmentation is the process of dividing a customer base into groups of individuals who share similar characteristics.  
This allows businesses to better understand their customers, personalize marketing strategies, improve engagement, and increase conversion rates.

In this project, we address the problem of **customer segmentation for a retail store** using **unsupervised machine learning**.  
Since no labels exist in the dataset, clustering techniques are used to discover hidden customer groups based on demographic and behavioral patterns.

The results of this analysis will be used to generate **actionable business recommendations** for targeted marketing and customer relationship management.

---

## ❓ Main Questions We Aim to Answer

To structure the analysis, this project is organized around **three key questions**:

---

- Who are our customers?
- What defines each customer segment?
- How do segments differ in spending, engagement, and responsiveness?
- How can marketing strategies be tailored to each segment?
- How do age, income, family structure, and time as a client vary across the population?
- What does spending behavior look like overall?


## 🎯 Project Objective

The main objectives of this project are to:

- Clean and prepare the data for analysis  
- Explore customer behavior and relationships between features  
- Reduce dimensionality for better visualization and clustering  
- Identify meaningful customer segments using unsupervised learning  
- Translate analytical findings into **business recommendations**

---

## 🧭 Methodology Overview

This analysis follows the full clustering workflow:

1. Data Cleaning & Feature Engineering  
2. Exploratory Data Analysis (Univariate, Bivariate, Multivariate)  
3. Encoding & Scaling  
4. Dimensionality Reduction (PCA, t-SNE)  
5. Clustering (K-Means)  
6. Cluster Profiling & Business Insights  

---

*By the end of this notebook, we will have a clear understanding of who the customers are, how they differ, and how the business can leverage these insights to improve marketing performance.*

## 📥 1. Data Loading & Environment Setup

```
Number of datapoints: 2240
Number of features: 29
```

<details>
<summary>📋 View output (click to expand)</summary>

```
     ID  Year_Birth   Education Marital_Status   Income  Kidhome  Teenhome  \
0  5524        1957  Graduation         Single  58138.0        0         0   
1  2174        1954  Graduation         Single  46344.0        1         1   
2  4141        1965  Graduation       Together  71613.0        0         0   
3  6182        1984  Graduation       Together  26646.0        1         0   
4  5324        1981         PhD        Married  58293.0        1         0   
5  7446        1967      Master       Together  62513.0        0         1   
6   965        1971  Graduation       Divorced  55635.0        0         1   
7  6177        1985         PhD        Married  33454.0        1         0   
8  4855        1974         PhD       Together  30351.0        1         0   
9  5899        1950         PhD       Together   5648.0        1         1   

  Dt_Customer  Recency  MntWines  ...  NumWebVisitsMonth  AcceptedCmp3  \
0  04-09-2012       58       635  ...                  7             0   
1  08-03-2014       38        11  ...                  5             0   
2  21-08-2013       26       426  ...                  4             0   
3  10-02-2014       26        11  ...                  6             0   
4  19-01-2014       94       173  ...                  5             0   
5  09-09-2013       16       520  ...                  6             0   
6  13-11-2012       34       235  ...                  6             0   
7  08-05-2013       32        76  ...                  8             0   
8  06-06-2013       19        14  ...                  9             0   
9  13-03-2014       68        28  ...                 20             1   

   AcceptedCmp4  AcceptedCmp5  AcceptedCmp1  AcceptedCmp2  Complain  \
0             0             0             0             0         0   
1             0             0             0             0         0   
2             0             0             0             0         0   
3             0             0             0             0         0   
4             0             0             0             0         0   
5             0             0             0             0         0   
6             0             0             0             0         0   
7             0             0             0             0         0   
8             0             0             0             0         0   
9             0             0             0             0         0   

   Z_CostContact  Z_Revenue  Response  
0              3         11         1  
1              3         11         0  
2              3         11         0  
3              3         11         0  
4              3         11         0  
5              3         11         0  
6              3         11         0  
7              3         11         0  
8              3         11         1  
9              3         11         0  

[10 rows x 29 columns]
```

</details>

### Features:

#### People

ID: Customer's unique identifier

Year_Birth: Customer's birth year

Education: Customer's education level

Marital_Status: Customer's marital status

Income: Customer's yearly household income

Kidhome: Number of children in customer's household

Teenhome: Number of teenagers in customer's household

Dt_Customer: Date of customer's enrollment with the company

Recency: Number of days since customer's last purchase

Complain: 1 if the customer complained in the last 2 years, 0 otherwise

#### Products

MntWines: Amount spent on wine in last 2 years

MntFruits: Amount spent on fruits in last 2 years

MntMeatProducts: Amount spent on meat in last 2 years

MntFishProducts: Amount spent on fish in last 2 years

MntSweetProducts: Amount spent on sweets in last 2 years

MntGoldProds: Amount spent on gold in last 2 years

#### Promotion

NumDealsPurchases: Number of purchases made with a discount

AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise

AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise

AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise

AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise

AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise

Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

#### Place

NumWebPurchases: Number of purchases made through the company’s website

NumCatalogPurchases: Number of purchases made using a catalogue

NumStorePurchases: Number of purchases made directly in stores

NumWebVisitsMonth: Number of visits to company’s website in the last month

---
## ⚡ 2. Memory Optimization
---
Data types were optimized to reduce memory footprint, improving efficiency for later transformations.

```
Memory usage of DataFrame before optimization is 0.50 MB
Memory usage after optimization is: 0.11 MB
Decreased by 78.0%
```

---

## 🔍 3. Initial Data Exploration

---

<details>
<summary>📋 View output (click to expand)</summary>

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2240 entries, 0 to 2239
Data columns (total 29 columns):
 #   Column               Non-Null Count  Dtype   
---  ------               --------------  -----   
 0   ID                   2240 non-null   int16   
 1   Year_Birth           2240 non-null   int16   
 2   Education            2240 non-null   category
 3   Marital_Status       2240 non-null   category
 4   Income               2216 non-null   float32 
 5   Kidhome              2240 non-null   int8    
 6   Teenhome             2240 non-null   int8    
 7   Dt_Customer          2240 non-null   category
 8   Recency              2240 non-null   int8    
 9   MntWines             2240 non-null   int16   
 10  MntFruits            2240 non-null   int16   
 11  MntMeatProducts      2240 non-null   int16   
 12  MntFishProducts      2240 non-null   int16   
 13  MntSweetProducts     2240 non-null   int16   
 14  MntGoldProds         2240 non-null   int16   
 15  NumDealsPurchases    2240 non-null   int8    
 16  NumWebPurchases      2240 non-null   int8    
 17  NumCatalogPurchases  2240 non-null   int8    
 18  NumStorePurchases    2240 non-null   int8    
 19  NumWebVisitsMonth    2240 non-null   int8    
 20  AcceptedCmp3         2240 non-null   int8    
 21  AcceptedCmp4         2240 non-null   int8    
 22  AcceptedCmp5         2240 non-null   int8    
 23  AcceptedCmp1         2240 non-null   int8    
 24  AcceptedCmp2         2240 non-null   int8    
 25  Complain             2240 non-null   int8    
 26  Z_CostContact        2240 non-null   int8    
 27  Z_Revenue            2240 non-null   int8    
 28  Response             2240 non-null   int8    
dtypes: category(3), float32(1), int16(8), int8(17)
memory usage: 111.7 KB
```

</details>

<details>
<summary>📋 View output (click to expand)</summary>

```
                      count          mean           std     min       25%  \
ID                   2240.0   5592.159821   3246.662198     0.0   2828.25   
Year_Birth           2240.0   1968.805804     11.984069  1893.0   1959.00   
Income               2216.0  52247.250000  25173.076172  1730.0  35303.00   
Kidhome              2240.0      0.444196      0.538398     0.0      0.00   
Teenhome             2240.0      0.506250      0.544538     0.0      0.00   
Recency              2240.0     49.109375     28.962453     0.0     24.00   
MntWines             2240.0    303.935714    336.597393     0.0     23.75   
MntFruits            2240.0     26.302232     39.773434     0.0      1.00   
MntMeatProducts      2240.0    166.950000    225.715373     0.0     16.00   
MntFishProducts      2240.0     37.525446     54.628979     0.0      3.00   
MntSweetProducts     2240.0     27.062946     41.280498     0.0      1.00   
MntGoldProds         2240.0     44.021875     52.167439     0.0      9.00   
NumDealsPurchases    2240.0      2.325000      1.932238     0.0      1.00   
NumWebPurchases      2240.0      4.084821      2.778714     0.0      2.00   
NumCatalogPurchases  2240.0      2.662054      2.923101     0.0      0.00   
NumStorePurchases    2240.0      5.790179      3.250958     0.0      3.00   
NumWebVisitsMonth    2240.0      5.316518      2.426645     0.0      3.00   
AcceptedCmp3         2240.0      0.072768      0.259813     0.0      0.00   
AcceptedCmp4         2240.0      0.074554      0.262728     0.0      0.00   
AcceptedCmp5         2240.0      0.072768      0.259813     0.0      0.00   
AcceptedCmp1         2240.0      0.064286      0.245316     0.0      0.00   
AcceptedCmp2         2240.0      0.013393      0.114976     0.0      0.00   
Complain             2240.0      0.009375      0.096391     0.0      0.00   
Z_CostContact        2240.0      3.000000      0.000000     3.0      3.00   
Z_Revenue            2240.0     11.000000      0.000000    11.0     11.00   
Response             2240.0      0.149107      0.356274     0.0      0.00   

                         50%       75%       max  
ID                    5458.5   8427.75   11191.0  
Year_Birth            1970.0   1977.00    1996.0  
Income               51381.5  68522.00  666666.0  
Kidhome                  0.0      1.00       2.0  
Teenhome                 0.0      1.00       2.0  
Recency                 49.0     74.00      99.0  
MntWines               173.5    504.25    1493.0  
MntFruits                8.0     33.00     199.0  
MntMeatProducts         67.0    232.00    1725.0  
MntFishProducts         12.0     50.00     259.0  
MntSweetProducts         8.0     33.00     263.0  
MntGoldProds            24.0     56.00     362.0  
NumDealsPurchases        2.0      3.00      15.0  
NumWebPurchases          4.0      6.00      27.0  
NumCatalogPurchases      2.0      4.00      28.0  
NumStorePurchases        5.0      8.00      13.0  
NumWebVisitsMonth        6.0      7.00      20.0  
AcceptedCmp3             0.0      0.00       1.0  
AcceptedCmp4             0.0      0.00       1.0  
AcceptedCmp5             0.0      0.00       1.0  
AcceptedCmp1             0.0      0.00       1.0  
AcceptedCmp2             0.0      0.00       1.0  
Complain                 0.0      0.00       1.0  
Z_CostContact            3.0      3.00       3.0  
Z_Revenue               11.0     11.00      11.0  
Response                 0.0      0.00       1.0
```

</details>

<details>
<summary>📋 View output (click to expand)</summary>

```
ID                      0
Year_Birth              0
Education               0
Marital_Status          0
Income                 24
Kidhome                 0
Teenhome                0
Dt_Customer             0
Recency                 0
MntWines                0
MntFruits               0
MntMeatProducts         0
MntFishProducts         0
MntSweetProducts        0
MntGoldProds            0
NumDealsPurchases       0
NumWebPurchases         0
NumCatalogPurchases     0
NumStorePurchases       0
NumWebVisitsMonth       0
AcceptedCmp3            0
AcceptedCmp4            0
AcceptedCmp5            0
AcceptedCmp1            0
AcceptedCmp2            0
Complain                0
Z_CostContact           0
Z_Revenue               0
Response                0
dtype: int64
```

</details>

```
np.int64(0)
```

![output](assets/output_001.png)

**Insight:**  
We observe that missing values in the *Income* column represent only **1.1%** of the data.  
Since this proportion is very small, removing these rows will not significantly affect the overall analysis, so we can safely drop them.

---
## 🧹 4. Data Cleaning
---

```
Number of unique values in Z_CostContact column: 1
Number of unique values in Z_Revenue column: 1
```

**Insight:**  
We also observe that the columns *Z_CostContact* and *Z_Revenue* each contain only **one unique value**.  
Since these features have no variability, they do not provide any useful information for analysis or modeling, and therefore we remove them from the dataset.

```
Newest customer's enrolment date is 2014-06-29
Oldest customer's enrolment date is 2012-07-30
```

---
## 🛠️ 5. Feature Engineering
---

Calculate the age of each customer

 Calculate the number of days each customer has been with the company

 Standardize 'Marital_Status' into 2 broader groups

Standardize 'Education' into 3 broader groups

 Combined the number of children and teenagers into a single feature to represent the total number of dependents in a household.

 Aggregated spending across all product categories to capture each customer’s total expenditure.

 Summed all accepted marketing campaigns to measure overall customer responsiveness to promotions.

 Combined all purchase channels to represent the total number of transactions made by each customer.

```
   Age     Education Marital_Status   Income  Kids  Days_is_client  Recency  \
0   56      Graduate         Single  58138.0     0             663       58   
1   59      Graduate         Single  46344.0     2             113       38   
2   48      Graduate        Partner  71613.0     0             312       26   
3   29      Graduate        Partner  26646.0     1             139       26   
4   32  Postgraduate        Partner  58293.0     1             161       94   

   Expenses  TotalNumPurchases  TotalAcceptedCmp  Complain  Response  
0      1617                 25                 0         0         1  
1        27                  6                 0         0         0  
2       776                 21                 0         0         0  
3        53                  8                 0         0         0  
4       422                 19                 0         0         0
```

After feature engineering and data cleaning, the original columns used to create new features were removed, and only the newly engineered features were retained to avoid redundancy and simplify the dataset.

---
## 🎯 6. Outlier Detection & Removal
---

```
                    count       mean        std     min      25%      50%  \
Age                2216.0     44.180     11.986    17.0     36.0     43.0   
Income             2216.0  52247.254  25173.076  1730.0  35303.0  51381.5   
Days_is_client     2216.0    353.521    202.435     0.0    180.0    355.5   
Recency            2216.0     49.013     28.948     0.0     24.0     49.0   
Expenses           2216.0    607.075    602.900     5.0     69.0    396.5   
TotalNumPurchases  2216.0     14.881      7.671     0.0      8.0     15.0   

                       75%       max  
Age                   54.0     120.0  
Income             68522.0  666666.0  
Days_is_client       529.0     699.0  
Recency               74.0      99.0  
Expenses            1048.0    2525.0  
TotalNumPurchases     21.0      44.0
```

```
                    count  mean  std    min    25%    50%    75%     max
Age                2216.0   0.0  1.0 -2.268 -0.683 -0.098  0.820   6.327
Income             2216.0   0.0  1.0 -2.007 -0.673 -0.034  0.647  24.413
Days_is_client     2216.0  -0.0  1.0 -1.747 -0.857  0.010  0.867   1.707
Recency            2216.0  -0.0  1.0 -1.693 -0.864 -0.000  0.863   1.727
Expenses           2216.0   0.0  1.0 -0.999 -0.893 -0.349  0.732   3.182
TotalNumPurchases  2216.0  -0.0  1.0 -1.940 -0.897  0.016  0.798   3.797
```

![output](assets/output_002.png)

![output](assets/output_003.png)

![output](assets/output_004.png)

![output](assets/output_005.png)

![output](assets/output_006.png)

![output](assets/output_007.png)

After plotting the boxplots, we observed the presence of outliers in the **Age** and **Income** features. To better understand their distributions and assess the impact of these extreme values, we proceeded to visualize their histograms.

![output](assets/output_008.png)

![output](assets/output_009.png)

![output](assets/output_010.png)

![output](assets/output_011.png)

![output](assets/output_012.png)

![output](assets/output_013.png)

In fact, both **Income** and **Age** show slight skewness in their distributions, which further indicates the presence of outliers and extreme values in these features.

---
## 🎯 7. Outlier Detection & Removal
---

To remove outliers, we first excluded values with a **z-score greater than 3** and then applied the **IQR method** to remove any remaining extreme values.

```
(18, 6)
```

---
## 📊 8. Univariate Analysis
---

After outlier removal, we can perform **univariate analysis** more cleanly, as the distributions are now more reliable and easier to interpret.

![output](assets/output_014.png)

![output](assets/output_015.png)

![output](assets/output_016.png)

![output](assets/output_017.png)

![output](assets/output_018.png)

![output](assets/output_019.png)

![output](assets/output_020.png)

![output](assets/output_021.png)

![output](assets/output_022.png)

![output](assets/output_023.png)

![output](assets/output_024.png)

![output](assets/output_025.png)

**Key Findings from Numerical Visualizations:**  

**Income and Age Distribution:** After removing outliers, income follows a roughly normal distribution, suggesting that most customers earn around the average income, with fewer customers earning significantly more or less.  

**Days as Client & Recency:** Both features exhibit a fairly uniform distribution, indicating that customers have been with the company for varying lengths of time and have interacted recently across a wide range of periods.  

**Expenses Distribution:** Expenses show an exponential distribution, meaning that most customers have lower spending, with spending rapidly decreasing as amounts increase.  

**Total Number of Purchases:** This feature follows a binomial-like distribution, reflecting common purchasing behaviors among customers, such as making a specific number of purchases.

![output](assets/output_026.png)

![output](assets/output_027.png)

![output](assets/output_028.png)

**Key Findings from Categorical and Binary Feature Visualizations:**  

**Education:** Most customers have a graduate-level education.  

**Kids:** The majority of customers have either no children or just one child.  

**Total Accepted Campaigns:** Most customers have never accepted any marketing offers.

![output](assets/output_029.png)

![output](assets/output_030.png)

![output](assets/output_031.png)

**Key Findings from Binary Feature Visualizations:**  

**Marital Status:** Most customers have a partner.  

**Complain:** The majority of customers have never lodged a complaint.

**Initial Customer Profile (Univariate Analysis):**  

Based on the distributions of numerical, categorical, and binary features:  
- Most customers are graduates.  
- Most have no kids or only one child.  
- Most have a partner.  
- The majority have never accepted marketing offers.  
- Most customers have never lodged a complaint.  

This provides an initial understanding of the customer base before proceeding to multivariate analysis and clustering.

## 📈 8. Bivariate Analysis

![output](assets/output_032.png)

![output](assets/output_033.png)

![output](assets/output_034.png)

![output](assets/output_035.png)

![output](assets/output_036.png)

![output](assets/output_037.png)

![output](assets/output_038.png)

![output](assets/output_039.png)

![output](assets/output_040.png)

![output](assets/output_041.png)

![output](assets/output_042.png)

![output](assets/output_043.png)

![output](assets/output_044.png)

![output](assets/output_045.png)

![output](assets/output_046.png)

**Customer Profiles by Education Level:**  

- **Graduate customers:** Most have one child or no children.  
- **Undergraduate customers:** Tend to have the fewest children overall.  
- **Postgraduate customers:** Most have either zero or one child.  
- **Offer acceptance:** Graduate and postgraduate customers rarely accept promotional offers.  
- **Marital status:** Most graduate and postgraduate customers have a partner.

---
## 🔢 9. Feature Encoding
---

**Feature Encoding:**  

- One-hot encoding was applied to categorical features because they have a limited number of categories. This allows the model to use these features effectively in clustering and analysis.

```
                               0        1        2        3        4
Age                         56.0     59.0     48.0     29.0     32.0
Income                   58138.0  46344.0  71613.0  26646.0  58293.0
Days_is_client             663.0    113.0    312.0    139.0    161.0
Recency                     58.0     38.0     26.0     26.0     94.0
Expenses                  1617.0     27.0    776.0     53.0    422.0
TotalNumPurchases           25.0      6.0     21.0      8.0     19.0
Complain                     0.0      0.0      0.0      0.0      0.0
Response                     1.0      0.0      0.0      0.0      0.0
Education_Undergraduate      0.0      0.0      0.0      0.0      0.0
Education_Postgraduate       0.0      0.0      0.0      0.0      1.0
Kids_1                       0.0      0.0      0.0      1.0      1.0
Kids_2                       0.0      1.0      0.0      0.0      0.0
Kids_3                       0.0      0.0      0.0      0.0      0.0
TotalAcceptedCmp_1           0.0      0.0      0.0      0.0      0.0
TotalAcceptedCmp_2           0.0      0.0      0.0      0.0      0.0
TotalAcceptedCmp_3           0.0      0.0      0.0      0.0      0.0
TotalAcceptedCmp_4           0.0      0.0      0.0      0.0      0.0
Marital_Status_Single        1.0      1.0      0.0      0.0      0.0
```

---
## 🌐 10. Multivariate Analysis
---

![output](assets/output_047.png)

**Pairplot Insights:**  

The pairplot of numerical features reveals the following patterns:  
- **Income and Total Number of Purchases:** Positive trend – customers with higher income tend to make more purchases.  
- **Expenses and Total Number of Purchases:** Positive trend – customers who spend more also tend to purchase more often.  
- **Income and Expenses:** Positive trend – higher-income customers generally have higher expenses.  

Overall, the scatterplots show clear linear relationships among these three variables, highlighting the impact of income on

![output](assets/output_048.png)

**Correlation Heatmap Insights:**  

The correlation heatmap confirms our observations from the pairplot, showing strong positive correlations among key features:  
- **Income & Expenses:** 0.83 – higher-income customers tend to spend more.  
- **Income & Total Number of Purchases:** 0.71 – higher-income customers make more purchases.  
- **Expenses & Total Number of Purchases:** 0.76 – customers who spend more also tend to purchase more frequently.  

These correlations highlight the close relationship between customer income, spending, and purchasing behavior.

![output](assets/output_049.png)

**Key Insights from Bivariate PairPlot Analysis:**  

1. **Income drives spending and purchases:** Higher-income customers spend more and buy more frequently, making income the most influential variable for customer value.  

2. **Expenses & Total Purchases are tightly linked:** Customers who purchase more also spend more; these features capture similar behavior from different angles.  

3. **Education impacts purchasing behavior:** Graduates and Postgraduates dominate higher income, expenses, and purchases, while Undergraduates are mostly lower across all three.  

4. **Age has limited effect:** Income-adjusted behavior is similar across age groups; age alone is not a strong predictor.  

5. **Recency and client duration matter less:** Recent customers or long-term clients do not consistently spend more; value depends on income and engagement rather than temporal factors.  

6. **Non-linear patterns exist:** Education classes overlap in 2D projections, suggesting non-linear methods may better separate groups.

![output](assets/output_050.png)

**Key Insights: Impact of Number of Kids on Customer Behavior**  

1. **Income decreases as number of kids increases:**  
Customers with 0 kids are concentrated in higher income ranges, 1–2 kids have moderate income, and 3 kids are mostly lower income.  

2. **Spending drops with more kids:**  
Expenses are highest for 0 kids and gradually decrease as the number of kids rises. Larger families have lower discretionary spending.  

3. **Purchase frequency declines with household size:**  
TotalNumPurchases is highest for 0 kids, slightly lower for 1 kid, and noticeably lower for 2–3 kids.  

4. **Income → Expenses → Purchases chain persists:**  
Regardless of kids, higher income leads to higher expenses and purchases; number of kids shifts customers along this curve rather than changing it.  

5. **Age is not a strong factor:**  
All kid categories appear across a wide age range; age alone does not explain family size or spending.  

6. **Recency and client duration independent of kids:**  
Days_is_client and Recency show similar distributions across all family sizes.  

7. **Segmentation potential:**  
- High-value: 0 kids, high income, high expenses, frequent purchases  
- Mid-value: 1–2 kids, moderate income and purchases  
- Low-value: 3 kids, low income and low spending  

**Summary:** Fewer kids → higher income, more spending, and more purchases; larger families spend less despite similar engagement duration.

![output](assets/output_051.png)

**Key Insights: Impact of Marital Status on Customer Behavior**  

1. **Partnered customers tend to have higher income:**  
Income distribution for customers with a partner is shifted toward higher values, while singles concentrate in lower to mid-income ranges.  

2. **Spending is higher for partnered customers:**  
Expenses are generally higher for those with a partner, likely due to dual income or shared financial responsibility.  

3. **Purchase frequency is higher among partnered customers:**  
TotalNumPurchases is higher on average for partnered customers; singles make fewer purchases.  

4. **Core income → spending → purchases relationships persist:**  
Marital status shifts customers along the same spending curve: higher income still leads to higher expenses and more purchases.  

5. **Age is not a strong differentiator:**  
Both singles and partnered customers span similar age ranges; age is secondary to income and household structure.  

6. **Engagement timing is independent of marital status:**  
Recency and Days_is_client show similar distributions for both groups; marital status does not affect interaction timing.  

7. **Segmentation potential:**  
- High-value: Partnered, higher income, higher expenses, frequent purchases  
- Low-value: Single, lower income, lower expenses, fewer purchases  

**Summary:** Customers with a partner earn more, spend more, and purchase more frequently, while recency and loyalty duration remain similar across marital statuses.

![output](assets/output_052.png)

**Key Insights: Customer Behavior by Education Level (Parallel Coordinates)**

1. **Income is the strongest differentiator:**  
Postgraduates consistently earn the most, Graduates are in the middle, and Undergraduates earn the least. Education strongly impacts earning power.

2. **Spending scales with education:**  
Expenses increase from Undergraduates → Graduates → Postgraduates, mirroring income levels. Higher education corresponds to higher spending capacity.

3. **Purchase frequency rises with education:**  
TotalNumPurchases is highest for Postgraduates, moderate for Graduates, and lowest for Undergraduates. Education affects both engagement and buying activity.

4. **Age is not a discriminating factor:**  
Lines overlap heavily across all education groups for Age. Differences in behavior are not explained by age once education is considered.

5. **Loyalty and recency are similar:**  
Days_is_client and Recency show strong overlap for all education levels. Education does not influence engagement timing.

6. **Overall customer profiles by education:**  
- Undergraduate → low income, low expenses, low purchases  
- Graduate → moderate income, moderate spending and activity  
- Postgraduate → high income, high expenses, high purchase frequency

**Summary:** Education primarily drives income, which in turn affects expenses and purchase behavior, while age and engagement duration remain largely unchanged.

![output](assets/output_053.png)

![output](assets/output_054.png)

![output](assets/output_055.png)

**Correlation Insights & Key Findings**

---

### 1️ Income, Expenses, Purchases
* Strong positive correlations:  
  - Income ↔ Expenses: 0.83  
  - Income ↔ TotalNumPurchases: 0.71  
  - Expenses ↔ TotalNumPurchases: 0.76  
* **Takeaway:** High-income customers spend more and buy more; these three variables capture core customer value.

---

### 2️ Campaign Response
* Response ↔ Expenses: 0.26  
* Response ↔ Income: 0.17  
* Response ↔ TotalNumPurchases: 0.15  
* Response ↔ Recency: –0.20  
* **Takeaway:** Customers who spend and buy more, and recent buyers, respond more to campaigns.

---

### 3️ Minor Influences
* Age, Days_is_client, Marital status, Complaints: weak correlations  
* Education & Kids: small effects  
* **Takeaway:** These features have limited impact on spending or response.

---

### 4️ Campaign Loyalty
* Past campaign responses positively correlated  
* **Takeaway:** Customers who respond once are more likely to respond again.

----
## 📐 11. Scaling & Dimensionality Reduction
---

We applied PCA for dimensionality reduction to simplify the data, improve model performance, and speed up computation. Beforehand, we scaled the data since PCA is sensitive to feature scaling

```
PCA()
```

![output](assets/output_056.png)

![output](assets/output_057.png)

From the scree plot, we chose to keep 10 components, which explain 80% of the cumulative variance—sufficient for the next analysis.

---
## 🗺️ 12. t-SNE Visualization
---

![output](assets/output_058.png)

![output](assets/output_059.png)

![output](assets/output_060.png)

From the t-SNE plots, we can faintly observe potential clusters in the data, though they are not very clear. We will proceed with clustering and then visualize the results to identify clearer groupings

---
## 🎨 13. Customer Segmentation Using K-Means
---

![output](assets/output_061.png)

Using the Elbow method, we evaluated different numbers of clusters (2–8) and observed that the curve starts to bend at 3 clusters. This suggests that 3 is an appropriate number of clusters for our KMeans analysis.

![output](assets/output_062.png)

After applying KMeans clustering, we visualized the clusters along with their centroids. The clusters are well-separated with very few overlapping points, indicating that the clustering is effective and meaningful.

![output](assets/output_063.png)

After clustering, the t-SNE projection shows much clearer and more distinct clusters, confirming that the segmentation captures meaningful differences between customer groups

---
## 👥 14. Cluster Profiling & Interpretation
---

```
                Age        Income  Days_is_client     Expenses  \
Clusters                                                         
0         46.452250  70542.718750      377.664105  1200.354555   
1         48.004866  43415.785156      328.885645   213.031630   
2         39.817654  35507.144531      338.777003   159.087108   

          TotalNumPurchases  TotalAcceptedCmp  Complain  Response  
Clusters                                                           
0                 21.547750          0.604830  0.004391  0.238200  
1                 11.727494          0.097324  0.017032  0.104623  
2                  9.307782          0.073171  0.010453  0.081301
```

![output](assets/output_064.png)

![output](assets/output_065.png)

![output](assets/output_066.png)

### Cluster Insights

---

#### Cluster 0 – High-Value Loyal Customers (VIPs)

**Profile & Demographics**
- **Age:** 45–60 (middle-aged to older)  
- **Income:** Very high (~65k–80k+)  
- **Customer tenure (Days_is_client):** Longest  
- **Family:** Mostly partnered, 0–1 kids  

**Behavior & Spending**
- **Total expenses:** High (median ~1,200–1,500)  
- **Purchase frequency:** Highest (median ~20–25)  
- **Shopping behavior:** Regular buyers, comfortable with premium pricing  

---

#### Cluster 1 – Mid-Value Growth Customers

**Profile & Demographics**
- **Age:** 40–55  
- **Income:** Medium (~40k–50k)  
- **Customer tenure:** Medium  
- **Family:** Mixed marital status, 1–2 kids  

**Behavior & Spending**
- **Total expenses:** Moderate (median ~200–250)  
- **Purchase frequency:** Moderate (median ~10–12)  
- **Outliers:** Some higher spenders   

---

#### Cluster 2 – Low-Value / Price-Sensitive Customers

**Profile & Demographics**
- **Age:** 30–45 (youngest cluster)  
- **Income:** Lowest (~25k–35k)  
- **Customer tenure:** Short  
- **Family:** Mostly single or 0–1 kids  

**Behavior & Spending**
- **Total expenses:** Low (median ~50–100)  
- **Purchase frequency:** Few (median ~5–10)  
- **Shopping behavior:** Price-sensitive, casual buyers

---
## 💼 15. Customer Segmentation – Business Strategies & Actions
---
Based on the cluster analysis, here are actionable strategies for each customer segment:

---

###  **Cluster 0 – High-Value Loyal Customers (VIPs)**
1. Launch VIP loyalty programs with exclusive rewards.
2. Offer premium bundles and upsell high-margin products.
3. Personalized communication (emails, messages) to maintain engagement.
4. Provide early access to new products or services.
5. Organize events or webinars targeting high-value customers.

---

###  **Cluster 1 – Mid-Value Growth Customers**
6. Send targeted promotions and discounts to encourage upsell.
7. Create bundle offers to increase average order value.
8. Educational campaigns highlighting product benefits.
9. Use personalized recommendations based on past purchases.
10. Incentivize repeat purchases through loyalty points or cashback.

---

###  **Cluster 2 – Low-Value / Price-Sensitive Customers**
11. Offer entry-level products or starter packages.
12. Run low-cost, automated engagement campaigns (email/SMS).
13. Provide limited-time discounts to attract purchases.
14. Introduce onboarding campaigns to increase brand familiarity.
15. Focus on cost-effective retention, avoid high-expense campaigns.


---

<div align="center">

### 🎓 About This Project

This project was developed as part of a data science exploration into **unsupervised learning** and **customer analytics**.
It demonstrates the end-to-end clustering workflow — from raw data to **actionable business strategy**.

<br>

**Made with 💜 by Gheffari Nour El Houda**

<sub>If you found this useful, consider giving the repo a ⭐</sub>

</div>
