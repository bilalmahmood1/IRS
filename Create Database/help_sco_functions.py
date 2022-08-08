# -*- coding: utf-8 -*-
"""
Helper functions to find the customers with SCO accounts
@author: BM387
"""

import pandas as pd
import numpy as np


# File containing the joint and individual account relations
sco_non_sco_codes_path = r"./data fixed/COINTESTAZIONI_V3.csv"
    
def get_sco_accounts(_id, mappings):
    """
    Returns the parent SCO account
    """
    
    for i in mappings:
        if _id in mappings[i]:
            return i
        
    
def get_members(df_mapping, ndg_sco):
    """
    Get members of the SCO account
    """
    return list(df_mapping[df_mapping["ndg_codificato"] == ndg_sco]["joint_codificato"].values)
    

def get_sco(df_mapping, ndg_cod):
    """
    Gets the SCO of the ndg
    """
    return list(df_mapping[df_mapping["joint_codificato"] == ndg_cod]["ndg_codificato"].values)
 
    
def sco_to_non_sco(df_mapping):
    """
    Returns the mapping between sco and non accoutns
    """
    all_scos =  set(df_mapping["ndg_codificato"])
    mappings = {}
    for sco in all_scos:
        mappings[sco] = get_members(df_mapping, sco)
    return mappings     

def get_sco_account_mappings():
    """
    Returns the mapping dictionary between SCO account and its individual customers inside it
    """
    df_sco_customer = pd.read_csv(sco_non_sco_codes_path, sep = ";")
    df_sco_customer.dropna(inplace = True)
    df_sco_customer["joint_codificato"] = df_sco_customer["joint_codificato"].apply(lambda x: np.int64(x))
    df_sco_customer.drop_duplicates(inplace = True)
    
    return sco_to_non_sco(df_sco_customer)
    