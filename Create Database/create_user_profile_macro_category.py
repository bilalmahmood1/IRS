# -*- coding: utf-8 -*-
"""
Build user profile macro for 2018 only. This is different because we got breakdown
for insurances for 2019 later on and for that different script was written

@author: BM387
"""

import pandas as pd

category = "RC tutela legale"
year = 2018

df_customers = pd.read_csv("./profile {}/targets {}.csv".format(year,category), sep = ";")



cols = list(df_customers.columns)
cols[0] = "ndg_codificato"
df_customers.columns = cols




#df_targets_previous = df_customers[["ndg_codificato","previous purchase"]]

df_anagrafica = pd.read_csv("./data fixed/anagrafica_profile.csv", 
                            sep = ";")


# Remove the ages below 10 and above 80 as these are not important in buying the product
df_anagrafica.drop(["(0, 10](AGE)", "(80, 90](AGE)", "(90, 100](AGE)", "(100, 110](AGE)", "(110, 120](AGE)", "(120, 130](AGE)", "(130, 140](AGE)"], 
                   axis = 1,
                   inplace = True)





# Removing Not important features and some of the logical features  
df_anagrafica.drop(['Commercial Property', 'Private Property', 
                    'joint account', 'FFF(GENDER)', "Unknown(CAE)"], 
                   axis = 1,
                   inplace = True)


# Removing Provinces
provinces = []
for i in df_anagrafica.columns:
    if "(PROVINCE)" in i:
        provinces.append(i)
df_anagrafica.drop(provinces, 
                   axis = 1,
                   inplace = True)



df_non_italaino = pd.read_csv("./data fixed/foreigner.csv", 
                            sep = ";")




df_members_profile = pd.read_csv("./profile {}/members.csv".format(year), 
                                   sep = ";")

df_controlling_profile = pd.read_csv("./profile {}/controlling.csv".format(year), 
                                   sep = ";")


df_salary_profile= pd.read_csv("./profile {}/salary.csv".format(year), 
                                   sep = ";")

df_spending_profile = pd.read_csv("./profile {}/spending.csv".format(year), 
                                   sep = ";")


df_max_spending_gain_profile = pd.read_csv("./profile {}/controlling_gain_loss.csv".format(year), 
                                   sep = ";")



df_historic_purchase = pd.read_csv("./profile {}/historic_purchase targets.csv".format(year), 
                                   sep = ";")




df_customer_profile = df_customers.merge(df_anagrafica, on = "ndg_codificato", how = "left")
df_customer_profile.dropna(inplace = True)

df_customer_profile = df_customer_profile.merge(df_members_profile, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_controlling_profile, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_salary_profile, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_spending_profile, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_max_spending_gain_profile, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_historic_purchase, on = "ndg_codificato", how = "left")


df_customer_profile["foreigner"] = df_customer_profile["ndg_codificato"].isin(df_non_italaino["ndg_codificato"])
df_customer_profile["foreigner"] = df_customer_profile["foreigner"].apply(lambda x: 1 if x else 0)
df_customer_profile[df_customer_profile["foreigner"] == 1]["purchase"].mean()


df_customer_profile.to_csv("./profile {}/customer profile {}.csv".format(year, category),
                           sep = ";", index = False)