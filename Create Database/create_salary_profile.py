# -*- coding: utf-8 -*-
"""
Create avergae monthly salary 6 months prior to the contact date for the accounts in test campaign, 2018 campaign and 2019 campaign 
@author: BM387
"""


from help_profiling_functions import *

years = ["test", 2018, 2019]
for year in years:

    # Target accounts to profile
    df_targets = pd.read_csv(r"./profile {}/targets.csv".format(year),
                                  sep = ";")

    # Creating average monthly salary profile
    df_salary_profile = make_salary_profile(df_targets, months = 6)

    # Saving the salary profile
    df_salary_profile.to_csv("./profile {}/salary.csv".format(year), 
                                       sep = ";",
                                       index = False)


