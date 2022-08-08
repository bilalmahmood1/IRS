# -*- coding: utf-8 -*-
"""
Built user profile by combining several created features ranging from demographics, wealth, salary, spending, members, historic purchases.
@author: BM387
"""

import pandas as pd

## Creating customer profiles for customer contacted in campaigns of 2018 and 2019
years = ["2018", "2019"]
for source in years:

  ## Reading the ndgs of the accounts contacted
  df_customers = pd.read_csv("./profile {}/targets.csv".format(source), sep = ";")
  cols = list(df_customers.columns)
  cols[0] = "ndg_codificato"
  df_customers.columns = cols

  ## Reading the anagrafica
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


  ## Reading the Foreigner data
  df_non_italaino = pd.read_csv("./data fixed/foreigner.csv", 
                              sep = ";")


  ## Reading the number of SCO members the account has 
  df_members_profile = pd.read_csv("./profile {}/members.csv".format(source), 
                                     sep = ";")

  ## Reading the wealth profile
  df_controlling_profile = pd.read_csv("./profile {}/controlling.csv".format(source), 
                                     sep = ";")

  ## Reading the salary profile
  df_salary_profile= pd.read_csv("./profile {}/salary.csv".format(source), 
                                     sep = ";")

  ## Reading the spending profile
  df_spending_profile = pd.read_csv("./profile {}/spending.csv".format(source), 
                                     sep = ";")

  ## Reading the max_min wealth change profile
  df_max_spending_gain_profile = pd.read_csv("./profile {}/controlling_gain_loss.csv".format(source), 
                                     sep = ";")

  ## Reading the historc purchase profile
  df_historic_purchase = pd.read_csv("./profile {}/historic_purchase targets.csv".format(source), 
                                     sep = ";")

  ## Joining all the profile features
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
  
  ## Saving customer profiles
  df_customer_profile.to_csv("./profile {}/customer profile.csv".format(source),
                             sep = ";", index = False)


## Combining the two campaigns to produce larger training dataset
df_customer_profile_2018 = pd.read_csv("./profile 2018/customer profile.csv", sep = ";")
df_customer_profile_2019 = pd.read_csv("./profile 2019/customer profile.csv", sep = ";")
df_customer_profile_combined = pd.concat([df_customer_profile_2018, df_customer_profile_2019]).copy()
## Saving campaigns 2018 + 2019
df_customer_profile_combined.to_csv("./profile 2018+2019/customer profile.csv".format(source),
                           sep = ";", index = False)
        
