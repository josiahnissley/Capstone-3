# Healthcare Cost Predictor – Neural Network

Objective: _Design a deep learning neural network to predict individual healthcare expenses_

__1. Data__
  
A useful dataset found on Kaggle containing 6+ features for model training https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance

__2. Data Wrangling__

[Data Wrangling Notebook]

The dataset was first checked for null values and duplicates, revealing no missing data and only one duplicate, which was dropped. Categorical features—'sex,' 'smoker,' and 'region'—were one-hot encoded, creating binary columns for 'sex' and 'smoker' and three new columns for 'region.' Numerical features were standardized using StandardScaler from sklearn to ensure consistency in scale. The cleaned and preprocessed dataset was then split into versions for exploratory data analysis and model training.

__3. EDA__

I began by examining correlations between features and the target variable, 'charges,' using a heatmap 

![image](https://github.com/user-attachments/assets/e3b7f61c-729e-4664-9999-832639a2ab73)

The top three strongest correlates were smoking status, age, and BMI. Scatterplots

![image](https://github.com/user-attachments/assets/23089d63-b04b-4c25-b26f-05b646dfd549)

further emphasized these relationships, highlighting that smokers experienced significantly higher charges, especially those with higher BMI. Additionally, trendlines showed charges increased at similar rates with age for smokers and non-smokers but rose more steeply for smokers with higher BMI.

To better understand the data distribution, histograms

![image](https://github.com/user-attachments/assets/cb9ae06e-f8f6-49b0-99f4-7ab3af04775f)

were created for age, BMI, and smoking status. The age distribution was relatively balanced, with a slight skew toward younger individuals. BMI followed a normal distribution centered around 30, and there were over twice as many non-smokers as smokers in the dataset. These distributions aligned with expectations, providing confidence to proceed to the modeling phase.

