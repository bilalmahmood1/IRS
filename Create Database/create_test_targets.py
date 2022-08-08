# -*- coding: utf-8 -*-
"""
This script selects all the non SCO accounts from the controlling data 
for January 2019, and adds the Data del Contatto of 2019-01-01, mimicking 
the scenario of running campaign on 1st January 2019
@author: BM387
"""


import pandas as pd
from help_sco_functions import *
import numpy as np


def extract_datetime(date_string):
    """Extract date from the contact date string"""
    date = str(date_string)
    day = int(date[6:])
    month = int(date[4:6])
    year = int(date[:4])
    return pd.datetime(year, month, day)


sco_dict = get_sco_account_mappings()
sco_accounts = list(sco_dict.keys())


# Selecting Month of January
month = 1
df_controlling_2019 = pd.read_csv("./data 2019/controlling.csv", sep = ";")
df_controlling_2019_month = df_controlling_2019[df_controlling_2019["mese"] == month]
df_controlling_2019_january_customers = list(set(df_controlling_2019_month["ndg_codificato"]))
accouts = len(df_controlling_2019_january_customers)


# Getting SCO accounts
df_sco = pd.read_csv("./data fixed/COINTESTAZIONI_V3.csv", sep = ";", 
                     usecols = ["joint_codificato"])


# Randomly shuffling the accounts
random_accounts = np.random.choice(df_controlling_2019_january_customers, 
                                   accouts, 
                                   replace = False)

# Just considering non-SCO accounts
target_customers  = [int(i) for i in random_accounts if int(i) not in sco_accounts]
target_customers = target_customers[0:accouts]


## Adding contact date
df_targets_2019 = pd.DataFrame(target_customers)
df_targets_2019.columns = ['NDG_cod']
df_targets_2019["Data del Contatto"] = "20190101"

# Saving the target customers
df_targets_2019.to_csv("./data test/test_campaign.csv", sep = ";", index = False)

