
import numpy as np
import pandas as pd
df = pd.DataFrame({
    'Country':['USA','China','India','Russia','Switzerland','Japan','Sweden','Singapore','South Korea','UK','Australia'],
    'Co2_emission':[5107.393,10877.218,2454.774,1764.866,39.738,1320.776,np.NaN,55.018,673.324,379.150,764.866],
    'Population_million':[329,1433,1366,145,8.5,126,10,5.8,51,67,np.NaN],
    'Continent': ['NA','Asia','Asia','EU','EU','Asia','EU','Asia','Asia','EU','AU']
                 })
                 
df.sort_values('Population_million',ascending = False)

	
df.sort_values('Population_million',ascending = False,na_position = 'first')


df.sort_values(by = ['Co2_emission','Population_million'],ascending = [True,False])


df.sort_values(by='Country')

df [ 'Continent'] = pd.Categorical(df['Continent'], categories=["NA", "Asia", "AU"

df.sort_values(by='Continent')



reorderlist = ['China','Russia','Australia','India','Japan','Singapore','South Korea','UK','USA','Sweden','Switzerland']
df.set_index('Country',inplace=True)
df.reindex(reorderlist)

	
df.sort_values('Population_million',ascending = False).reset_index(drop=True)

df['Date']=pd.date_range(start='1/12/2019', end='1/22/2019', freq='D')

df.sort_values('Date',ascending=False,inplace=True)

df.set_index('Date').sort_index()

df = pd.DataFrame(data={'x':[28,10,90], 'y':[45,58,67],'z':[19,82,37]}, index=['a', 'b', 'c'])
df.sort_values(by='a', axis=1)

#using numpy
df.iloc[:, np.argsort(df.loc['a'])]
df.apply(np.sort, axis = 1)

f[['x', 'y', 'z']] = np.sort(df)[:, ::-1] 
df.reindex(sorted(df.columns, key=lambda x: df[x]['a']), axis=1)

