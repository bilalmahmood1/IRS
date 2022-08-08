# -*- coding: utf-8 -*-
"""

Creates for all target accounts in test campaign, 2018 campaign and 2019 campaign whether they bought any insurance prior to contact date
@author: BM387
"""

from help_profiling_functions import *

years = ["test", 2018, 2019]

for year in years:

    # Target accounts to profile
    df_targets = pd.read_csv("./profile {}/targets.csv".format(year), sep = ";")


    # Insurnaces traces
    df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                                  parse_dates = ["start date","end date","emission date"],
                                  sep =";")

        
    ## Adds whether accounts previously bought any insurance
    previous_sparkasse_purchased(df_assicurativo, df_targets) 

    df_targets_results = df_targets[["ndg_contraente","previous purchase"]]

    cols = list(df_targets_results.columns)
    cols[0] =  "ndg_codificato"
    df_targets_results.columns = cols

    # Saving the historic purchaseprofiles
    df_targets_results.to_csv("./profile {}/historic_purchase targets.csv".format(year), index = False, sep = ";")
