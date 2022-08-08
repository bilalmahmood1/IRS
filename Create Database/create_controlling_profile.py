# -*- coding: utf-8 -*-
"""
Create Controlling profile for the users specified in the targets(in test campaign, 2018 campaign and 2019 campaign):

"ndg_codificato","individual debit","individual stock",
"individual(debit + stock)", "family debit","family stock","family(debit + stock)"

@author: BM387
"""

from help_profiling_functions import *


## Years for which to profile the data
years = ["test", 2018, 2019]
for year in years:   

    ## Accessing target customer list
    df_targets = pd.read_csv(r"./profile {}/targets.csv".format(year),
                                  sep = ";")

    ## Making countrolling profile for each account
    df_controlling_profile = make_controlling_profile(df_targets, months = 6)

    ##Saving the controlling profile
    df_controlling_profile.to_csv("./profile {}/controlling.csv".format(year), 
                                       sep = ";",
                                       index = False)



