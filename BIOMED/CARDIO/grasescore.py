import pandas as pd
from gracescore import GraceScore 

data=pd.read_csv("sample_data.csv")
Grace=GraceScore()

data['grace_in_hospital']=data.apply(lambda row: Grace.CalculateAdmissionToInHospitalDeath(row['age'],row['EkgHR'], row['SystolicPressure'], row['Creatinine'],row['KillipClass'],row['CardiacArrest'],row['Troponin'] ,row['EkgSTdeviation']),axis=1)

data['grace180']=data.apply(lambda row: Grace.CalculateSixMonthDeath(row['age'],row['EkgHR'], row['SystolicPressure'], row['Creatinine'],row['KillipClass'],row['CardiacArrest'],row['Troponin'] ,row['EkgSTdeviation']),axis=1)

data['grace365']=data.apply(lambda row: Grace.CalculateOneYearDeath(row['age'],row['EkgHR'], row['SystolicPressure'], row['Creatinine'],row['KillipClass'],row['CardiacArrest'],row['Troponin'] ,row['EkgSTdeviation']),axis=1)

