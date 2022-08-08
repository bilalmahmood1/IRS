# -*- coding: utf-8 -*-
"""
Helper functions to make all the essential account profiles defined in the problem
@author: BM387
"""


import pandas as pd
from extract_features import *
from help_sco_functions  import *
from extract_features import *
from datetime import timedelta
from pandas import Timestamp



## Getting the SCO account and individual account mappings
mappings = get_sco_account_mappings()


def make_controlling_profile(df_targets, months = 6):
    """
    Given the targets, query the database to extract the controlling information, calculates the average monthly 
    "individual debit","individual stock", "individual(debit + stock)", "family debit","family stock","family(debit + stock) 6 months prior to the 
    contact date.

    output dataframe: "individual debit","individual stock", "individual(debit + stock)", "family debit","family stock","family(debit + stock)
    """
    
    results = []
    
    for row in df_targets.iterrows():
        print(row[0])
        ndg = row[1]["ndg_contraente"]
        purchase_date = pd.Timestamp(row[1]["contact date"])
               
        days = months * 30
        start_date = purchase_date + pd.Timedelta('-{} days'.format(days))
        
        start_date = start_date.strftime("%Y-%m-%d")
        purchase_date = purchase_date.strftime("%Y-%m-%d")
        
        
        ndg_controlling =  get_controlling_information(ndg, start_date, purchase_date, months)
        sco_account = get_sco_accounts(ndg, mappings)
    
        # if the account has other accounts in sco relation, add their controlling effects as family controlling effects      
        if sco_account:
            members = mappings[sco_account]
            family_controlling = np.array(get_controlling_information(sco_account, start_date, purchase_date, months))
                
            for member in members:
                if member != ndg:
                    member_controlling =  get_controlling_information(member, start_date, purchase_date, months)
                    family_controlling = family_controlling + np.array(member_controlling)             
                else:
                    family_controlling = family_controlling + np.array(ndg_controlling)
                    
            family_controlling = list(family_controlling)
            
        else:
            family_controlling = ndg_controlling
            
        results.append([ndg] + ndg_controlling + family_controlling)
            
    return pd.DataFrame(results, columns = ["ndg_codificato","individual debit","individual stock",
                                   "individual(debit + stock)", "family debit","family stock","family(debit + stock)"])
    
    


def make_maximum_gain_spending_profile(df_targets, months = 12):
    """
    Given the targets, query the controlling database to extract
    maximum gain in wealth or maximum fall in wealth(spending) month to month wealth 12 months prior to the purchase date.

    output dataframe: "ndg_codificato","individual wealth decrease","individual wealth increase",
                                   "family wealth decrease","family wealth increase"
    """
    
    results = []
    
    for row in df_targets.iterrows():
        print(row[0])
        ndg = row[1]["ndg_contraente"]
        purchase_date = pd.Timestamp(row[1]["purchase date"])
               
        # 30 days buffer added
        days = months * 30
        start_date = purchase_date + pd.Timedelta('-{} days'.format(days))
        
        start_date = start_date.strftime("%Y-%m-%d")
        purchase_date = purchase_date.strftime("%Y-%m-%d")
        
        
        ndg_controlling =  get_gain_spending_information(ndg, start_date, purchase_date)
        
        sco_account = get_sco_accounts(ndg, mappings)
        
        ## Adding the effects of other SCO members if there are any
        if sco_account:
            members = mappings[sco_account]
            family_controlling = np.array(get_gain_spending_information(sco_account, start_date, purchase_date))
                
            for member in members:
                if member != ndg:
                    member_controlling =  get_gain_spending_information(member, start_date, purchase_date)
                    family_controlling = family_controlling + np.array(member_controlling)             
                else:
                    family_controlling = family_controlling + np.array(ndg_controlling)
                    
            family_controlling = list(family_controlling)
            
        else:
            family_controlling = ndg_controlling
            
        results.append([ndg] + ndg_controlling + family_controlling)
            
        
        
    return pd.DataFrame(results, columns = ["ndg_codificato","individual wealth decrease","individual wealth increase",
                                   "family wealth decrease","family wealth increase"])
    


def previous_sparkasse_purchased(df, df_contact):
    """
    Returns whether the customer bought something before the contact date - appox 31.
        Returns
        (1, most recent emission date) if present in the insurance purchase table before the contact date(df)
        (0, 0)
    """
    
    previous_purchases = []
    previous_purchase_date = []
        
    for t in df_contact.iterrows():
        margin_days = 31
        ndg = t[1]["ndg_contraente"]
        contact_date = pd.Timestamp(t[1]["contact date"])
  
        df_result = df[(df["ndg_contraente"] == ndg) 
                             & (df["emission date"] <= (contact_date - timedelta(days=margin_days)))]
        
        if df_result.shape[0] > 0:
            
            previous_purchases.append(1)
            previous_purchase_date.append(df_result["emission date"].max())
            
        else:
            
            previous_purchases.append(0)
            previous_purchase_date.append("")
            
            
    df_contact["previous purchase date"] = previous_purchase_date
    df_contact["previous purchase"] = previous_purchases
    
    
def make_salary_profile(df_targets, months = 6):
    """Given the targets, query the database to extract the average salary
    information 6 months prioir to the contact date

    output dataframe: "ndg_codificato","individual salary", "family salary"
    """
    
    results = []
    
    for row in df_targets.iterrows():
        print(row[0])
        ndg = row[1]["ndg_contraente"]
        purchase_date = pd.Timestamp(row[1]["contact date"])
               
        days = months * 30
        start_date = purchase_date + pd.Timedelta('-{} days'.format(days))
        
        start_date = start_date.strftime("%Y-%m-%d")
        purchase_date = purchase_date.strftime("%Y-%m-%d")
        
        
        ndg_salary =  get_salary_information(ndg, start_date, purchase_date, months)
        sco_account = get_sco_accounts(ndg, mappings)
        
        if sco_account:
            members = mappings[sco_account]
            family_salary = np.array(get_salary_information(sco_account, start_date, purchase_date, months))
            for member in members:
                if member != ndg:
                    member_salary =  get_salary_information(member, start_date, purchase_date, months)
                    family_salary = family_salary + np.array(member_salary)             
                else:
                    family_salary = family_salary + np.array(ndg_salary)
                              
            family_salary = list(family_salary)
            
        else:
            family_salary = ndg_salary
            
        results.append([ndg] + ndg_salary + family_salary)
            
    return pd.DataFrame(results, columns = ["ndg_codificato","individual salary", "family salary"])




def make_spending_profile(df_targets, months = 6):
    """
    Given the targets, query the database to extract the average spending
    information 6 months prioir to the contact date
    """
    
    results = []
    
    for row in df_targets.iterrows():
        print(row[0])
        ndg = row[1]["ndg_contraente"]
        purchase_date = pd.Timestamp(row[1]["contact date"])
               
        days = months * 30
        start_date = purchase_date + pd.Timedelta('-{} days'.format(days))
        
        start_date = start_date.strftime("%Y-%m-%d")
        purchase_date = purchase_date.strftime("%Y-%m-%d")
        
        
        ndg_spending =  get_spending_information(ndg, start_date, purchase_date, months)
        sco_account = get_sco_accounts(ndg, mappings)
        
        if sco_account:
            members = mappings[sco_account]
            family_spending = get_spending_information(sco_account, start_date, purchase_date, months)
            for member in members:
                if member != ndg:
                    member_spending =  get_spending_information(member, start_date, purchase_date, months)
                    family_spending = family_spending + member_spending           
                else:
                    family_spending = family_spending + ndg_spending
                              
            family_spending = family_spending
            
        else:
            family_spending = ndg_spending
            
        results.append([ndg] + list(ndg_spending.values.reshape(-1)) + list(family_spending.values.reshape(-1)))
            
    return pd.DataFrame(results, columns = ["ndg_codificato"] 
                        + [i + "-individual" for i in list(ndg_spending.index)]
                        + [i + "-family" for i in list(family_spending.index)])
    
    
