import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Question 1)1)
data = pd.read_csv('data.txt',sep='\t')
df = data[data['horsepower']!="?"]
df['horsepower'] = pd.to_numeric(df['horsepower'])

#df.plot.scatter(x='horsepower',y='mpg', color = 'black')

#Question 2)i)
mpg_y = df.loc[:,'mpg'].values.reshape(-1,1)
horsepower_x = df.loc[:,'horsepower'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(horsepower_x,mpg_y)
mpg_pred = reg.predict(horsepower_x)

#Question 2)ii)
print('Coefficient: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
print('Mean squared error: \n', mean_squared_error(mpg_pred, mpg_y))
print('Coefficient of determination: \n', reg.score(horsepower_x, mpg_y))

#Question 2)iii)
#plt.plot(horsepower_x, mpg_pred, color='red')

horsepower_x_sm = sm.add_constant(horsepower_x)
model = sm.OLS(mpg_y, horsepower_x_sm).fit()
mpg_pred_sm = model.predict(horsepower_x_sm)

#Question 2)ii)
print(model.summary())

#Plot of regression line using statsmodel library, same results as using sklearn
'''
plt.plot(horsepower_x, mpg_pred_sm, '-', color='blue')
plt.show()
'''

#Question 2)iv)
y_hat = reg.coef_*98 + reg.intercept_
print('Prediction of mpg with horsepower of 98: ', y_hat)

ci_lower = y_hat - 1.96*0.717
ci_upper = y_hat + 1.96*0.717
print("Confidence interval is: [", ci_lower, ", ", ci_upper, "]")

stdev = np.sqrt(sum((mpg_pred - mpg_y)**2) / (len(mpg_y)-2))
#stdev = np.sqrt(1/(len(mpg_y)-2) * sum((mpg_pred - mpg_y)**2))

pi_lower = y_hat - 1.96 * stdev
pi_upper = y_hat + 1.96 * stdev
print("Prediction interval is: [", pi_lower, ", ", pi_upper, "]")

#Question 3)3)
rss = np.sum(np.square(mpg_pred - mpg_y))
rse = math.sqrt(rss/((len(mpg_pred))-2))
#rse = 4.90575691954594

print(rse/np.mean(mpg_y))
#percentage error is 0.20923714066914834 about 21%

#Question 4
print(df['mpg'].median())
df['mpg01'] = np.where(df['mpg'] > df['mpg'].median(),1,0)


fig, ax = plt.subplots()
ax = sns.boxplot(y="cylinders", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs cylinders.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="displacement", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs displacement.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="horsepower", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs horsepower.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="acceleration", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs acceleration.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="year", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs year.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="origin", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs origin.png')

fig, ax = plt.subplots()
ax = sns.boxplot(y="weight", x="mpg01", data=df)
fig.savefig('Boxplot mpg01 vs weight.png')

df_train, df_test = train_test_split(df, test_size=0.3)
predictors = ['horsepower', 'cylinders', 'weight', 'displacement']
x_train = df_train[predictors]
y_train = df_train['mpg01']
x_test = df_test[predictors]
y_test = df_test['mpg01']

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
log_pred = logisticRegr.predict(x_train)
#fig, axs = plt.subplots(2, 2)
#sns.regplot(x=x_train[['horsepower']], y=log_pred, data=df_train, logistic=True)
#sns.regplot(x=x_train[['cylinders']], y=log_pred, data=df_train, logistic=True)
#sns.regplot(x=x_train[['weight']], y=log_pred, data=df_train, logistic=True)
#sns.regplot(x=x_train[['displacement']], y=log_pred, data=df_train, logistic=True)

predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)
print("Accuracy of the model is: ", score*100)

conf_mat = metrics.confusion_matrix(y_test, predictions)
print(conf_mat)

fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()