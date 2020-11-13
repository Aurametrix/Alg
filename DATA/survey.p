import pandas as pd

df = pd.read_csv('survey_results_public.csv')
df.head()
df.shape
# multiple choice
df['BetterLife'].value_counts()
df['BetterLife'].value_counts(normalize=True)
df['MgrMoney'].value_counts(normalize=True)
df['SocialMedia'].value_counts().plot(kind="bar")
df['SocialMedia'].value_counts().plot(kind="bar", figsize=(15,7), color="#61d199")

said_no = df[df['BetterLife'] == 'No']
said_no.head(3)

print(said_no['Age'].mean(),
      said_yes['Age'].mean(),
      said_no['Age'].median(),
      said_yes['Age'].median()
     )
     
filtered_1 = df[(df['BetterLife'] == 'Yes') & (df['Country'] == 'India')]

filtered = df[(df['BetterLife'] == 'Yes') & (df['Age'] >= 50) & (df['Country'] == 'India') &~ (df['Hobbyist'] == "Yes") &~ (df['OpenSourcer'] == "Never")]
filtered

python_bool = df["LanguageWorkedWith"].str.contains('Python')
python_bool.value_counts(normalize=True)
lang_df.stack().value_counts().plot(kind='bar', figsize=(15,7), color="#61d199")
