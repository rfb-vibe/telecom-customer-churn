# Customer Churn

According to [Profitwell](https://www.profitwell.com/customer-retention/industry-rates), the average churn rate within the telecommunications industry is 22% (churn rate referring to the rate customers close their accounts or end their business relationship).

# Business Understanding

The telecommunications company, SyriaTel, is faced with the problem of better predicting when its customers will soon churn. They need a solution that will predict whether a customer will ("soon") stop doing business with SyriaTel. This will be valuable to SyriaTel, so that they may better understand their churn rate and identify areas they may address to improve its churn rate.

Finding predictable patterns using a classification model will benefit SyriaTel's business practices to minimize customer churn.

To determine which classification model best predicts potential customer churn, I will evaluate models' performance using the F1 score. The F1 score is the harmonic average of two other metrics, precision and recall, and is suited well to evaluate imbalanced datasets.

Precision summarizes the fraction of examples assigned the positive class that belong to the positive class whereas the recall summarizes how well the positive class was predicted and is the same calculation as sensitivity. Both precision and recall values fall in the range [0,1], with 0 indicating no precision/recall and 1 perfect precision/recall. These values can be combined into one metric, the F1 score, which is the harmonic average of the precision and recall scores. The F1 score also ranges [0,1].

The closer to 1 the F1 score, the more perfect the model is classifying samples.

# Data Understanding

The data source for this project comes from [SyriaTel's churn data](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). This data is suitable for the project because it includes key performance indicators and data points from SyriaTel related to its customers and their accounts as well as whether the customer churned or not.

The data consists of 3,333 observations with 21 features and no missing values.

**Explanation of Features**

* `state`: the state the user lives in
* `account length`: the number of days the user has this account
* `area code`: the code of the area the user lives in
* `phone number`: the phone number of the user
* `international plan`: true if the user has the international plan, otherwise false
* `voice mail plan`: true if the user has the voice mail plan, otherwise false
* `number vmail messages`: the number of voice mail messages the user has sent
* `total day minutes`: total number of minutes the user has been in calls during the day
* `total day calls`: total number of calls the user has done during the day
* `total day charge`: total amount of money the user was charged by the Telecom company for calls during the day
* `total eve minutes`: total number of minutes the user has been in calls during the evening
* `total eve calls`: total number of calls the user has done during the evening
* `total eve charge`: total amount of money the user was charged by the Telecom company for calls during the evening
* `total night minutes`: total number of minutes the user has been in calls during the night
* `total night calls`: total number of calls the user has done during the night
* `total night charge`: total amount of money the user was charged by the Telecom company for calls during the night
* `total intl minutes`: total number of minutes the user has been in international calls
* `total intl calls`: total number of international calls the user has done
* `total intl charge`: total amount of money the user was charged by the Telecom company for international calls
* `customer service calls`: number of customer service calls the user has done
* `churn`: true if the user terminated the contract, otherwise false

# Modeling

I create vanilla models using a decision tree classifier, logistic regression, k-Nearest Neighbors classifier, random forest classifier, and eXtreme Gradient Boost (XGBoost) classifier.

Each model has its own advantages and disadvantages, which is why I will include each to best determine the strongest predictive model for the stakeholder.
 
