# -*- coding: utf-8 -*-
"""
Create maximum gain and maximum decrease in account profile in the test campaign, 2018 campaign and 2019 campaign by looking at month-over-month wealth changes 
of the accounts. The script queries the database, to calculate the required features

@author: BM387
"""
from help_profiling_functions import *

years = ["test", 2018, 2019]

for year in years:

    ## Get target customer list
    df_targets = pd.read_csv(r"profile {}\targets.csv".format(year),
                                  sep = ";")
      
    ## Building customer account wealth change profiles
    df_controlling_profile = make_maximum_gain_spending_profile(df_targets, months = 12)


    # Saving the wealth change profile
    df_controlling_profile.to_csv("./profile {}/controlling_gain_loss.csv".format(year), 
                                       sep = ";",
                                       index = False)


