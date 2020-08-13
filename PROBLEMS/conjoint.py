import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.DataFrame()
#read the input csv 
rank_data = pd.read_csv("/Users/prajwalsreenivas/Downloads/bike_conjoint.csv")
data  = rank_data
rank_data[['gear','type','susp','guards']] = rank_data['Attribute'].str.split(',',expand=True)
rank_data = rank_data.drop(columns=['Attribute'])
rank_data

#introduce dummy variables 
conjoint_data = pd.get_dummies(rank_data,columns =['gear','type','susp','guards'])
conjoint_data

#renaming columns of dataframe
fullNames = {"Rank":"Rank", \
           "gear_Has gears": "geared","gear_No gears": "fixedgear","type_ Mountain bike": "Mountain bike", \
          "type_ Racing bike": "Road Bike", "susp_ No Suspension":"Hardtail",  "susp_ Suspension":"Softtail", \
           "guards_ Mudguards":"Mudguarded", "guards_ No mudguards":"openmudguard"
          }

conjoint_data.rename(columns=fullNames, inplace=True)

X = conjoint_data[[ u'geared', u'fixedgear', u'Mountain bike', u'Road Bike',
       u'Hardtail', u'Softtail', u'Mudguarded', u'openmudguard']]
X = sm.add_constant(X)
Y = conjoint_data.Rank
linearRegression = sm.OLS(Y, X).fit()
linearRegression.summary()

conjoint_attributes = [u'geared', u'fixedgear', u'Mountain bike', u'Road Bike',
       u'Hardtail', u'Softtail', u'Mudguarded', u'openmudguard']
level_name = []
part_worth = []
part_worth_range = []
end = 1
for item in conjoint_attributes:
    nlevels = len(list(set(conjoint_data[item])))
    level_name.append(list(set(conjoint_data[item])))
    begin = end
    end = begin + nlevels - 1
    new_part_worth = list(linearRegression.params[begin:end])
    new_part_worth.append((-1) * sum(new_part_worth))
    part_worth_range.append(max(new_part_worth) - min(new_part_worth))
    part_worth.append(new_part_worth)
    # end set to begin next iteration

attribute_importance = []
for item in part_worth_range:
    attribute_importance.append(round(100 * (item / sum(part_worth_range)),2))


effect_name_dict = {u'geared':u'geared', u'fixedgear':u'fixedgear', u'Mountain bike':u'Mountain bike', u'Road Bike':u'Road Bike',
       u'Hardtail':u'Hardtail', u'Softtail':u'Softtail', u'Mudguarded':u'Mudguarded', u'openmudguard':u'openmudguard'}


#print out parthworth's for each level
estimates_of_choice = []
index = 0 
for item in conjoint_attributes : 
    print ("\n Attribute : " , effect_name_dict[item])
    print ("\n Importance : " , attribute_importance[index])
    print('    Level Part-Worths')
    for level in range(len(level_name[index])):
        print('       ',level_name[index][level], part_worth[index][level])
    index = index + 1
#calculating Utilities
flattened_list = [val for sublist in utility for val in sublist]
y = pd.Series(flattened_list)
df2=conjoint_data[[ u'geared', u'fixedgear', u'Mountain bike', u'Road Bike',
       u'Hardtail', u'Softtail', u'Mudguarded', u'openmudguard']]
df2 = df2.astype(float)
i=0
for item in conjoint_attributes:
    df2[item]= df2[item]*flattened_list[i]
    i=i+1
df2

utility_scores = df2.values.sum(axis=1)
max_utility = np.argmax(utility_scores)
print "The index of combination combination with hightest sum of utility scores is " 
print data.ix[max_utility]

total_utility=0
c= 0.833
for item in utility_scores:
    total_utility = total_utility + np.exp(c*item)

for item in utility_scores:
    probabilty = np.exp(c*item)/total_utility
    itemindex = np.where(utility_scores==item)

    print 'Market share of profile %s is %s ' % (itemindex,probabilty*100)
    
    
