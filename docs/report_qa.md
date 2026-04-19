Project: Fraud Detection Project
Submitted on: April 19, 2026


### QUESTIONS & ANSWERS


**Q1. Which insights did you gain from your EDA?**
During EDA we find that our dataset has 11 columns and 6 million+ bank transactions.
After doing univariate, bivariate, and multivariate analysis, we find that we have better chance when we combine with other features.
We also find that individual correlations with isFraud were all low, combinations of these features showed revealed much stronger fraud signals than any individual feature alone. 
These are the features that helped us guide our next step to preprocessing our data:
- type:
    - were able to narrow our focus to CASH_OUT and TRANSFER, eliminating the remaining three transaction types from fraud consideration.
- amount: 
    - the average transaction amount for fraud is about 8x higher than legitimate transactions. 
    - This is especially highlighted in CASH_OUT transactions where fraud and non-fraud amounts separate more clearly than in TRANSFER.
- newBalanceOrig: 
    - was the most predictive out of all the balances features, almost all of fraud cases had zero balance after a transaction.
- oldBalanceOrg:
    - fraudulent transactions almost always started with money in the account, only 0.5% of fraud cases had a zero balance before the transaction.


**Q2. How did you determine which columns to drop or keep? If your EDA informed this process, explain which insights you used to determine which columns were not needed.**
    
Here are some of the features we did not find very useful.
- step:
    - showed no consistent pattern with isFraud, some spikes in fraud rate shown in plots were attributed to low transaction volume which contradicts itself
- nameOrig & nameDest: 
    - these are unique strings used to identify each transactions, no learnable pattern for the model.
- isFlaggedFraud:
    - only 16 out of 8,213 fraud cases were caught, making it not a good predictor.

During preprocessing, 1,156 duplicate rows were identified after dropping the account ID columns nameOrig and nameDest. However, upon inspection, dropping these duplicates reduced our fraud transactions from 8,213 to 8,124 - a loss of 89 fraudulent transactions. Given the severe class imbalance in this dataset (only 0.13% of transactions are fraudulent), every fraud case is valuable for model training. Therefore, retaining these duplicate rows to preserve as many fraud examples as possible was the idea the project went with.

Dropping these duplicates will be a future exploration when optimizing the models.

Also engineered three new features (balance_delta_orig, balance_delta_dest, and balance_mismatch) to directly capture the fraud patterns identified in EDA

After creating these new features, we dropped the original balances columns (newBalanceOrig, oldBalanceOrg, newBalanceDest, oldBalanceDest) because they were already captured by the new engineered features.


**Q3. Which hyperparameter tuning strategy did you use? Grid-search or random-search? Why?**
We used Random-Search over Grid-Search primarily due to the size of the data. Grid-Search would have exhaustively tried every single combination of hyperparameter which would would have taken too long to find something. Random-Search on the other hand simply picks a fixed number of combinations and averages the F1 score out, which will end up finding a near-optimal solution.

**Q4. How did your model's performance change after discovering optimal hyperparameters?**
The hyperparameter tuning did not have a huge impact on the model's performance. Using the best parameters: *n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=1, and class_weight=None*, the baseline model did slightly better than the tuned model by 0.005. This seems backwards but this is explained by the fact that the tuned model is more honest than a single train/test split. Cross Validation split the data into 5 different ways and averages the score, which makes the tuned model's F1 Score more reliable.

The model's precision is really good, 96% correct when the model says a transaction is fraud, but recall is where it struggles - specifically missing 423 or 25.75% of fraudulent transactions. But some improvements could be done in handling the class imbalance using SMOTE or higher class weights. 

**Q5. What was your final F1 Score?**

|  | Score 
| ------ | ------ 
| Baseline F1-Score | 0.8488
| Tuned F1-Score | 0.8437
| AUC | 0.9972


Our model's precision is 96%, meaning that when it flags a transaction as fraud, it is right almost every time, very few false alarms. However, the model misses 423 actual fraud cases, which is 25.75% of all real fraud in our test set, a low recall. To improve this, future work could apply SMOTE to generate more synthetic fraud examples for training, or increase class weights to make the model pay more attention to fraud cases. Both would help catch more fraud, though some precision would likely be lost in the process.
The AUC score of 0.9972 means that the model is actually very good at identifying which transactions look suspicious. The issue is that by default, the model only flags something as fraud if it is at least 50% confident. Some of the 423 missed fraud cases are likely sitting just below that 50% cutoff.
By lowering the decision threshold from 0.50 to 0.30, recall improved from 74.25% to 77% while precision only dropped slightly from 96% to 94%. This simple adjustment required no retraining and resulted in a slightly better F1 score of 0.8462.
In short, the F1 score tells us how the model performs at the 50% cutoff, while the AUC tells us how good the model is at ranking risk overall. The adjustment we made on the threshold also confirms that tuning it is a quick and effective way to optimize the precision-recall tradeoff for imbalanced datasets.

