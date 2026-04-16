# Financial Fraud Detection Project

Fraud happens quietly, often when and where you least expect it — and that subtlety is exactly what makes detection challenging. If we _flag everything_ as fraud, we drown in false alarms but if we _don't flag anything_ as fraud, we’ll miss every real case. The challenge comes from finding the rare suspicious cases hiding in a sea of normal activities.

In this project, we’re assigned a role as data scientist at Caishen, an international bank in NYC, and the cybersecurity team has handed us historical fraud data. Now it’s our job to turn it into a minimal viable product (MVP), analyzing the dataset of bank transactions then building an ensemble classifier (such as random forest or boosted model) that can decide whether a transaction is likely to be fraudulent.


Information on the the data features are in this link: [Features Information](docs/features_info.txt)
Since Github blocks file uploads larger than 100 MB, we will not be including the actual dataset within this repository. But you can see the first few rows of the data here: [Caishen Bank Transactions Dataset](data/bank_transactions.png)

This project is split into three sections:
1. EDA
During EDA we find that our dataset has 11 feature and 6 million+ bank transactions and that these features aren't predictive enough on their own. After doing univariate, bivariate, and multivariate analysis, we find that we have better chance when we combine with other features.

These are the features that helped us guide our next step to preprocessing our data:
- **type:** 
    - were able to narrow our focus to CASH_OUT and TRANSFER, eliminating the remaining three transaction types from fraud consideration.
- **amount:** 
    - the average transaction amount for fraud is about 8x higher than real transactions. This is especially highlighted in CASH_OUT transactions where fraud and non-fraud amounts separate more clearly than in TRANSFER.
- **newBalanceOrig:** 
    - was the most predictive out of all the balances features, almost all of fraud cases had zero balance after a transaction.
- **oldBalanceOrg:** 
    - very few fraudulent transactions had no balance in accounts

We also find that individual correlations with isFraud were all low, combinations of these features showed a little bit more predictive value. On the other hand, here are some of the features we did not find very useful.
- **step:** 
    - no consistent pattern
- **nameOrig & nameDest:** 
    - just unique strings to ID the transactions
- **isFlaggedFraud:** 
    - only 16 out of 8,213 fraud cases were flagged, not very helpful



2. Cleaning, Wrangling, and Preprocessing



3. Creating and Hypertuning Model









---

To conclude this project:

iii. Which hyperparameter tuning strategy did you use? Grid-search or random-search? Why? 


iv. How did your model's performance change after discovering optimal hyperparameters? 


v. What was your final F1 Score? 
