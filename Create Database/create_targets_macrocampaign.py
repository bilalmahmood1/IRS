# -*- coding: utf-8 -*-
"""
Prepare Macro targets for 2019 campaign, combine 2018 individual ones, and add macro categories to test campaign
@author: BM387
"""

import pandas as pd

# Looking at campaign 2019 breakdown
df_targets_2019 = pd.read_csv("./profile 2019/targets.csv", sep = ";")



# New insurance breakdown file
df_breakdown = pd.read_excel("./assicurativo/breakdown campagna 2019.xlsx",
                             parse_dates = ["DT_EFFETTO"],
                             nrows = 30802)


# Align with the campaign 2019
df_breakdown = df_breakdown[df_breakdown["DT_EFFETTO"] > pd.datetime(2019,9,1)]
df_breakdown = df_breakdown[df_breakdown["DT_EFFETTO"] < pd.datetime(2020,9,1)]


# Looking at which category of insurance was purchased
df_targets_2019["Property"] = 0
df_targets_2019["RC tutela legale"] = 0
df_targets_2019["Infortunio"] = 0
df_targets_2019["Malattia"] = 0


for i in df_targets_2019.index:
    ndg = df_targets_2019.loc[i,"ndg_contraente"]
    person = df_breakdown[df_breakdown["ndg_codificato"] == ndg]
    purchases = person["macro category"]
    date = person["DT_EFFETTO"].min()
    
    for p in purchases:
        if not pd.isna(p):
            df_targets_2019.at[i, p] = 1
            df_targets_2019.at[i, "purchase"] = 1
            df_targets_2019.at[i, "purchase date"] = date
            


df_targets_2019.to_csv("./profile 2019/targets.csv", sep = ";", index = False ) 



# Campaign 2018 macro category alignement
df_any = pd.read_csv("./profile 2018/targets.csv", sep = ";")
df_property = pd.read_csv("./profile 2018/targets property.csv", sep = ";")
df_malattia = pd.read_csv("./profile 2018/targets malattia.csv", sep = ";")
df_infortunio = pd.read_csv("./profile 2018/targets infortunio.csv", sep = ";")
df_rc = pd.read_csv("./profile 2018/targets RC tutela legale.csv", sep = ";")
df_merge = df_any.merge(df_property[["ndg_contraente","purchase"]], on = "ndg_contraente")
df_merge = df_merge.merge(df_rc[["ndg_contraente","purchase"]], on = "ndg_contraente")
df_merge = df_merge.merge(df_infortunio[["ndg_contraente","purchase"]], on = "ndg_contraente")
df_merge = df_merge.merge(df_malattia[["ndg_contraente","purchase"]], on = "ndg_contraente")
df_merge.columns = list(df_any.columns) + ["Property", "RC tutela legale","Infortunio","Malattia"]

# 2018 macro category breakdown
df_merge.to_csv("./profile 2018/targets.csv", sep = ";", index = False ) 





# Test Campaign category breakdown
df_targets_test = pd.read_csv("./profile test/customer profile.csv", sep = ";")

df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                              parse_dates = ["start date","end date","emission date"],
                              sep =";")
 

# Focusing on 6 months after the contact
df_assicurativo_filter = df_assicurativo[df_assicurativo["start date"] > pd.datetime(2019,1,1)]
df_assicurativo_filter = df_assicurativo_filter[df_assicurativo_filter["start date"] < pd.datetime(2019,7,1)]


df_targets_test ["Property"] = 0
df_targets_test ["RC tutela legale"] = 0
df_targets_test ["Infortunio"] = 0
df_targets_test ["Malattia"] = 0

for i in df_targets_test.index:
    
    ndg = df_targets_test.loc[i,"ndg_codificato"]
    person = df_assicurativo_filter[df_assicurativo_filter["ndg_contraente"] == ndg]
    purchases = person["category"]
    date = person["start date"].min()
    
    for p in purchases:
        if not pd.isna(p):
            df_targets_test.at[i, p] = 1
            df_targets_test.at[i, "purchase"] = 1
            df_targets_test.at[i, "purchase date"] = date
            

df_targets_test.to_csv("./profile test/customer profile.csv", 
                       sep = ";", index = False ) 




