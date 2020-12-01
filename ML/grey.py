# ... load small set of the dataset in "df"
import seaborn as sns
sns.pairplot(df, hue="Neurodegenerative Disease", palette="husl")

import seaborn as sns
sns.heatmap(df.corr(), annot=True)

import numpy as np

# source: https://programmer.group/5e4b66268a670.html
# full code here: https://github.com/rsarai/grey-relational-analysis

class GreyRelationalCoefficient():

    def __init__(self, data, tetha=0.5, standard=True):
        '''
        data: Input matrix, vertical axis is attribute name, first column is parent sequence
        theta: Resolution coefficient, range 0~1，Generally take 0.5，The smaller the correlation coefficient is, the greater the difference is, and the stronger the discrimination ability is
        standard: Need standardization
        '''
        self.data = np.array(data)
        self.tetha = tetha
        self.standard = standard

    def get_calculate_relational_coefficient(self, parent_column=0):
        self.normalize()
        return self._calculate_relational_coefficient(parent_column)

    def _calculate_relational_coefficient(self, parent_column):
        momCol = self.data[:,parent_column].copy()
        sonCol = self.data[:,0:]

        for col in range(sonCol.shape[1]):
            sonCol[:,col] = abs(sonCol[:,col]-momCol)

        minMin = np.nanmin(sonCol)
        maxMax = np.nanmax(sonCol)

        cors = (minMin + (self.tetha * maxMax)) / (sonCol + (self.tetha * maxMax))
        return cors
Running and checking results:

# source: https://programmer.group/5e4b66268a670.html
# full code here: https://github.com/rsarai/grey-relational-analysis

K = len(df.columns)
correl = []
for i in range(K):
    model = GreyRelationalCoefficient(df, standard=True)
    cors = model.get_calculate_relational_coefficient(parent_column=i)
    mean_cors = cors.mean(axis=0)
    correl.append(mean_cors)

sns.heatmap(correl, annot=True, xticklabels=df.columns, ytickl
