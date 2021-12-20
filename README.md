# card-default
Build a machine learning model to predict if there will be a card default

# Amex Credit Card Default

# Approach

## EDA
- I first performed exploratory data analysis (EDA) on the train dataset

#### Observations
- Age spread is pretty even, except for extreme bins
- More females are issued credit cards than males, however males are likely to default more than women
- XNA gender count (1) is negliglible
- Customers having 4 or more children are outliers
- Spread of credit limit is even in terms of default
- Most people are provided credit limit in range $20-30k
- credit_limit_used seems to be a critical feature. 97% of customers who defaulted consumed greater than 70% of their credit limit 
- Customers with credit score less than 700 are highly likely to default
- Low-skill labour staff are more likely to default
- Customers with lesser employment experience are more likely to default
- Customers with higher credit limit are less likely to default. To finetune on this, credit score can be used as a filter to further investigate of default chances
- Customers who have used less than 70% of their limit do not default, regardless of credit limit
- Customers with higher debt payments to yearly income are likely to default
- Customers to default payments in past 6 months have defaulted on the credit bill

## Tools used for feature engineering and modelling
- Code for the problem is solved using Python programming language
- For EDA, I used visualization packages: matplotlib and seaborn 
- For data wrangling: pandas, numpy packages were used 
- For data modelling: imblearn, scikit-learn, xgboost

## Imputing missing values and creating new features
-   Initial list of features: 
>  ['customer_id', 'name', 'age', 'gender', 'owns_car', 'owns_house',
   'no_of_children', 'net_yearly_income', 'no_of_days_employed',
   'occupation_type', 'total_family_members', 'migrant_worker',
   'yearly_debt_payments', 'credit_limit', 'credit_limit_used(%)',
   'credit_score', 'prev_defaults', 'default_in_last_6months',
   'credit_card_default']  

- There were certain features which had missing values. The missing values for such features were replaced as follows:

    - `owns_car`, `owns_house` are filled with value `N`; then all features' missing cells are filled with 0
    
    - Using `credit_score` and `credit_limit_used(%)` below new features were created:
    
        - `above_min_credlim_def`: If the customer has used their credit limit (%) more than the lowest credit limit usage (%) among defaulted customers
        - `below_min_credscore_def`: If the customer's credit score is lower than the highest credit score among defaulted accounts
        - `above_min_credlim_occ`: If the customer has used their credit limit (%) more than the lowest credit limit usage (%) among defaulted customer sharing the same occupation
        - `below_min_credscore_occ`: If the customer's credit score is lower than the highest credit score among defaulted account sharing the same occupation
        - `credlim_to_income`: Credit limit to net yearly income of customer
        - `debt_to_income`: Yearly debt payment to net yearly income of customer
        
## Modelling

- For the modelling, an XGB classifier was used
- The parameters of the XGB classifier was tuned with the help of RandomizedSearchCV using 5 fold cross-validation with `objective: binary_logistic` and `scoring_metric: f1_macro`
-  The best parameters were used to train the final model
-  The output probabilites were converted to binary outputs using an appropriate threshold (0.73)
