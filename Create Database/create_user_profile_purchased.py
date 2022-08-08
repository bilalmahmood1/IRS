
# -*- coding: utf-8 -*-
"""
Build user profile for customers who purchased irrespected of contacted
@author: BM387
"""

import pandas as pd


df_customers_2018_bought = pd.read_csv("./profile 2018/purchase_targets.csv", sep = ";")

cols = list(df_customers_2018_bought.columns)
cols[0] = "ndg_codificato"
df_customers_2018_bought.columns = cols


df_customers_2018_bought["purchase"] = 1


df_targets = df_customers_2018_bought[["ndg_codificato","purchase"]]




df_targets_previous = df_customers_2018[["ndg_codificato","previous purchase"]]

df_anagrafica = pd.read_csv("./data fixed/anagrafica_profile.csv", 
                            sep = ";")

# Remove the ages below 10 and above 80 as these are not important in buying the product
df_anagrafica.drop(["(0, 10](AGE)", "(80, 90](AGE)", "(90, 100](AGE)", "(100, 110](AGE)", "(110, 120](AGE)", "(120, 130](AGE)", "(130, 140](AGE)"], 
                   axis = 1,
                   inplace = True)


# Removing Not important features     
df_anagrafica.drop(['Commercial Property', 'Private Property', 
                    'joint account', 'FFF(GENDER)', "Unknown(CAE)",
                    "Zona Industriale Bolzano(BRANCH)"], 
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




df_members_profile_2018 = pd.read_csv("./profile 2018/members.csv", 
                                   sep = ";")

df_controlling_profile_2018 = pd.read_csv("./profile 2018/controlling purchases.csv", 
                                   sep = ";")


df_salary_profile_2018 = pd.read_csv("./profile 2018/salary purchases.csv", 
                                   sep = ";")

df_spending_profile_2018 = pd.read_csv("./profile 2018/spending purchases.csv", 
                                   sep = ";")



df_customer_profile = df_targets.merge(df_targets_previous, on = "ndg_codificato", how = "left")

df_customer_profile = df_targets.merge(df_anagrafica, on = "ndg_codificato", how = "left")
df_customer_profile.dropna(inplace = True)

df_customer_profile = df_customer_profile.merge(df_members_profile_2018, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_controlling_profile_2018, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_salary_profile_2018, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_spending_profile_2018, on = "ndg_codificato", how = "left")






"""
# Removing Branch 
for i in df_anagrafica.columns:
    if "BRANCH" in i:
        branch.append(i)

        
df_anagrafica.drop(branch, 
                   axis = 1,
                   inplace = True)



# Removing Individual 
individuals = []
for i in df_spending_profile_2018.columns:
    if "individual" in i:
        individuals.append(i)



df_spending_profile_2018.drop(individuals, 
                   axis = 1,
                   inplace = True)

"""  




        

       
        

df_customer_profile.to_csv("./profile 2018/purchase customer profile.csv",sep = ";", index = False)






df_customers_2019 = pd.read_csv("./profile 2019/targets.csv", sep = ";")
cols = list(df_customers_2019.columns)
cols[0] = "ndg_codificato"
df_customers_2019.columns = cols
df_targets = df_customers_2019[["ndg_codificato","purchase"]]


df_targets_previous = df_customers_2019[["ndg_codificato","previous purchase"]]



df_anagrafica = pd.read_csv("./data fixed/anagrafica_profile.csv", 
                            sep = ";")


# Remove the ages below 10 and above 80 as these are not important in buying the product
df_anagrafica.drop(["(0, 10](AGE)", "(80, 90](AGE)", "(90, 100](AGE)", "(100, 110](AGE)", "(110, 120](AGE)", "(120, 130](AGE)", "(130, 140](AGE)"], 
                   axis = 1,
                   inplace = True)


# Removing Not important features     
df_anagrafica.drop(['Commercial Property', 'Private Property', 
                    'joint account', 'FFF(GENDER)', "Unknown(CAE)",
                    "Zona Industriale Bolzano(BRANCH)"], 
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




df_members_profile_2019 = pd.read_csv("./profile 2019/members.csv", 
                                   sep = ";")

df_controlling_profile_2019 = pd.read_csv("./profile 2019/controlling.csv", 
                                   sep = ";")


df_salary_profile_2019 = pd.read_csv("./profile 2019/salary.csv", 
                                   sep = ";")

df_spending_profile_2019 = pd.read_csv("./profile 2019/spending.csv", 
                                   sep = ";")




df_customer_profile = df_targets.merge(df_targets_previous, on = "ndg_codificato", how = "left")


df_customer_profile = df_customer_profile.merge(df_anagrafica, on = "ndg_codificato", how = "left")
df_customer_profile.dropna(inplace = True)

df_customer_profile = df_customer_profile.merge(df_members_profile_2019, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_controlling_profile_2019, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_salary_profile_2019, on = "ndg_codificato", how = "left")
df_customer_profile = df_customer_profile.merge(df_spending_profile_2019, on = "ndg_codificato", how = "left")



df_customer_profile.to_csv("./profile 2019/customer profile.csv",sep = ";", index = False)
