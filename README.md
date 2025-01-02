# Healthcare Cost Predictor – Neural Network

Objective: _Design a deep learning neural network to predict individual healthcare expenses_

__1. Data__ 
  
A useful dataset found on Kaggle containing 6+ features for model training https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance

__2. Data Wrangling & Preprocessing__

[Data Wrangling Notebook](https://github.com/josiahnissley/Predicting-Healthcare-Cost-Deep-Learning-/blob/main/1%20-%20Data_Wrangling.ipynb)

[Preprocessing Notebook](https://github.com/josiahnissley/Predicting-Healthcare-Cost-Deep-Learning-/blob/main/2%20-%20Preprocessing.ipynb)

The dataset was first checked for null values and duplicates, revealing no missing data and only one duplicate, which was dropped. Categorical features—'sex,' 'smoker,' and 'region'—were one-hot encoded, creating binary columns for 'sex' and 'smoker' and three new columns for 'region.' Numerical features were standardized using StandardScaler from sklearn to ensure consistency in scale. The cleaned and preprocessed dataset was then split into versions for exploratory data analysis and model training.

__3. EDA__

[EDA Notebook](https://github.com/josiahnissley/Predicting-Healthcare-Cost-Deep-Learning-/blob/main/3%20-%20EDA.ipynb)

I began by examining correlations between features and the target variable, 'charges,' using a heatmap 

![image](https://github.com/user-attachments/assets/e3b7f61c-729e-4664-9999-832639a2ab73)

The top three strongest correlates were smoking status, age, and BMI. Scatterplots

![image](https://github.com/user-attachments/assets/23089d63-b04b-4c25-b26f-05b646dfd549)

further emphasized these relationships, highlighting that smokers experienced significantly higher charges, especially those with higher BMI. Additionally, trendlines showed charges increased at similar rates with age for smokers and non-smokers but rose more steeply for smokers with higher BMI.

To better understand the data distribution, histograms

![image](https://github.com/user-attachments/assets/cb9ae06e-f8f6-49b0-99f4-7ab3af04775f)

were created for age, BMI, and smoking status. The age distribution was relatively balanced, with a slight skew toward younger individuals. BMI followed a normal distribution centered around 30, and there were over twice as many non-smokers as smokers in the dataset. These distributions aligned with expectations, providing confidence to proceed to the modeling phase.

__4. Modeling__

[Modeling Notebook](https://github.com/josiahnissley/Predicting-Healthcare-Cost-Deep-Learning-/blob/main/4%20-%20Modeling.ipynb)

I first built a baseline regression model using sklearn’s Random Forest Regressor with an 80/20 train-test split and 100 estimators. It achieved a Mean Absolute Error (MAE) of 2,934.18 and an R² of 0.80, explaining 80% of the variance in charges.

Next, I developed three neural network (NN) models with different architectures, each using the ReLU activation function, Adam optimizer, and MSE loss function. The best NN had 50, 100, 50 nodes and achieved an MAE of 3,055.17 and R² of 0.79, performing slightly worse than the baseline. 

![image](https://github.com/user-attachments/assets/2cc82e67-2075-420f-87c3-405f14057174)

I then optimized the NN using Keras Hyperband, which recommended 72, 120, 40 nodes and a learning rate of 0.001. This model improved to an MAE of 2,808.88 but showed only a minor improvement over previous models.

__Limitations__

As shown in the below figure

![image](https://github.com/user-attachments/assets/fcd325f4-05dc-4bdc-86ba-3a08fa429b06)

the models fit well for charges under $15,000, where data was more abundant, but struggled for higher charges due to fewer samples.

__Feature Importance__

Finally, this last figure 

![image](https://github.com/user-attachments/assets/635b51bd-bc08-4d24-b4d1-d32380c5ab9d)

confirms that smoking, age, and BMI were the top predictors, reinforcing earlier correlation findings.

__Application & Future Work__

The models demonstrated reasonable predictive capabilities, with mean absolute errors (MAE) around $3,000, effectively estimating healthcare costs within a ±$3,000 range. Given charges spanning $1,000 to $65,000, this level of accuracy is useful for preliminary cost estimation in healthcare.

To enhance performance, future efforts could expand the dataset—Increasing sample size, especially for higher charges, could improve predictions by providing better representation of outliers. Additionally, testing alternative models like XGBoost or Random Forest with advanced hyperparameter tuning may yield better results than neural networks. Implementing these strategies could further reduce errors and enhance predictive accuracy.

[Final Project PDF](https://github.com/josiahnissley/Predicting-Healthcare-Cost-Deep-Learning-/blob/main/Capstone%20III%20Report.pdf)

