# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:00:03 2019

@author: patel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
"""
Import Files
"""
columns = ['Physician_Profile_ID','Total_Amount_of_Payment_USDollars',
           'Date_of_Payment', 'Number_of_Payments_Included_in_Total_Amount',
           'Nature_of_Payment_or_Transfer_of_Value', 'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID', 'Record_ID']
chunks = pd.read_csv(r'OP_DTL_GNRL_PGYR2018_P06282019.csv', usecols=columns, chunksize=1000000)
data = pd.concat(chunks)

plt.style.use('seaborn')

"""
FOOD AND BEVERAGE: 

"""

foodbev_train = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Food and Beverage') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] != 100000011066)].sample(n = 100000, random_state = 123)
foodbev_test = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Food and Beverage') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] == 100000011066)]

# Transform and scale data
train = np.array(foodbev_train).astype(np.float)
test = np.array(foodbev_test).astype(np.float)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled =scaler.transform(test)

clustno = range(1, 10) 
kmeans = [KMeans(n_clusters=i) for i in clustno]
score = [kmeans[i].fit(train_scaled).score(train_scaled) for i in range(len(kmeans))]

# Plot Elbow Curve Results
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve (Food and Beverage)')
plt.show()
# Define the k-means model and fit to the data
kmeans = KMeans(n_clusters=5, random_state=123).fit(train_scaled)


foodbev_test['test_clusters'] = kmeans.predict(test_scaled)
test_clusters_centers = kmeans.cluster_centers_
foodbev_test['dist'] = [np.linalg.norm(x-y) for x, y in zip(test_scaled, test_clusters_centers[foodbev_test['test_clusters']])]

# Create fraud predictions based on outliers on clusters 
foodbev_test['km_pred'] = foodbev_test['dist'].apply(lambda x: 1 if x >= np.percentile(foodbev_test['dist'], 90) else 0)
foodbev_results = pd.merge(foodbev_test, data, left_index=True, right_index=True)
 
fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(111)

ax1.scatter(train_scaled[:,0],train_scaled[:,1], c='lightseagreen', marker="o", edgecolors = 'w', label='Training Data')
ax1.scatter(test_scaled[:,0],test_scaled[:,1], c='tomato', marker="o", edgecolors = 'w', label='Insys Data')
leg = plt.legend(loc='upper right', prop={'size': 15}, frameon = True)
leg.get_frame().set_edgecolor('black')
plt.title("Food and Beverage", fontsize = 18)
plt.ylabel("Payment Amount (Scaled)", fontsize = 16, labelpad=20)
plt.xlabel("Number of Payments (Scaled)", fontsize = 16, labelpad=20) 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


"""
Compensation for services other than consulting, including serving as faculty or as a speaker at a venue other than a continuing education program: 

"""

services_train = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Compensation for services other than consulting, including serving as faculty or as a speaker at a venue other than a continuing education program') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] != 100000011066)].sample(n = 100000, random_state = 123)
services_test = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Compensation for services other than consulting, including serving as faculty or as a speaker at a venue other than a continuing education program') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] == 100000011066)]

# Transform and scale data
train = np.array(services_train).astype(np.float)
test = np.array(services_test).astype(np.float)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled =scaler.transform(test)

clustno = range(1, 10) 
kmeans = [KMeans(n_clusters=i) for i in clustno]
score = [kmeans[i].fit(train_scaled).score(train_scaled) for i in range(len(kmeans))]

# Plot Elbow Curve Results
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve (Services)')
plt.show()
# Define the k-means model and fit to the data
kmeans = KMeans(n_clusters=4, random_state=123).fit(train_scaled)

services_test['test_clusters'] = kmeans.predict(test_scaled)
test_clusters_centers = kmeans.cluster_centers_
services_test['dist'] = [np.linalg.norm(x-y) for x, y in zip(test_scaled, test_clusters_centers[services_test['test_clusters']])]

# Create fraud predictions based on outliers on clusters 
services_test['km_pred'] = services_test['dist'].apply(lambda x: 1 if x >= np.percentile(services_test['dist'], 90) else 0)
services_results = pd.merge(services_test, data, left_index=True, right_index=True)

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(111)

ax1.scatter(train_scaled[:,0],train_scaled[:,1], c='lightseagreen', marker="o", edgecolors = 'w', label='Training Data')
ax1.scatter(test_scaled[:,0],test_scaled[:,1], c='tomato', marker="o", edgecolors = 'w', label='Insys Data')
leg = plt.legend(loc='upper right', prop={'size': 15}, frameon = True)
leg.get_frame().set_edgecolor('black')
plt.title("Services Other Than Consulting", fontsize = 18)
plt.ylabel("Payment Amount (Scaled)", fontsize = 16, labelpad=20)
plt.xlabel("Number of Payments (Scaled)", fontsize = 16, labelpad=20) 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()



"""
Consulting Fee: 

"""

consultingfee_train = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Consulting Fee') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] != 100000011066)].sample(n = 100000, random_state = 123)
consultingfee_test = data[['Number_of_Payments_Included_in_Total_Amount','Total_Amount_of_Payment_USDollars']].loc[(data['Nature_of_Payment_or_Transfer_of_Value'] == 'Consulting Fee') & (data['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID'] == 100000011066)]

# Transform and scale data
train = np.array(consultingfee_train).astype(np.float)
test = np.array(consultingfee_test).astype(np.float)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

clustno = range(1, 10) 
kmeans = [KMeans(n_clusters=i) for i in clustno]
score = [kmeans[i].fit(train_scaled).score(train_scaled) for i in range(len(kmeans))]

# Plot Elbow Curve Results
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve (Consulting Fee)')
plt.show()
# Define the k-means model and fit to the data
kmeans = KMeans(n_clusters=5, random_state=123).fit(train_scaled)

consultingfee_test['test_clusters'] = kmeans.predict(test_scaled)
test_clusters_centers = kmeans.cluster_centers_
consultingfee_test['dist'] = [np.linalg.norm(x-y) for x, y in zip(test_scaled, test_clusters_centers[consultingfee_test['test_clusters']])]

# Create fraud predictions based on outliers on clusters 
consultingfee_test['km_pred'] = consultingfee_test['dist'].apply(lambda x: 1 if x >= np.percentile(consultingfee_test['dist'], 90) else 0)
consultingfee_results = pd.merge(consultingfee_test, data, left_index=True, right_index=True)




fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(111)

ax1.scatter(train_scaled[:,0],train_scaled[:,1], c='lightseagreen', marker="o", edgecolors = 'w', label='Training Data')
ax1.scatter(test_scaled[:,0],test_scaled[:,1], c='tomato', marker="o", edgecolors = 'w', label='Insys Data')
leg = plt.legend(loc='upper right', prop={'size': 15}, frameon = True)
leg.get_frame().set_edgecolor('black')
plt.title("Consulting Fee", fontsize = 18)
plt.ylabel("Payment Amount (Scaled)", fontsize = 16, labelpad=20)
plt.xlabel("Number of Payments (Scaled)", fontsize = 16, labelpad=20) 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

outlierfoodandbev = foodbev_results['Record_ID'].loc[foodbev_results['km_pred'] == 1]
outlierservices = services_results['Record_ID'].loc[services_results['km_pred'] == 1]
outlierconsulting = consultingfee_results['Record_ID'].loc[consultingfee_results['km_pred'] == 1]

alloutliers = [outlierfoodandbev,outlierservices,outlierconsulting]
alloutliers = pd.concat(alloutliers)