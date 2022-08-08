
# -*- coding: utf-8 -*-
"""
Prepare Property, Malattia, Infortunio and RC targets for the 2018 campaign only.
@author: BM387
"""

import pandas as pd
from datetime import timedelta
from pandas import Timestamp

df_contacts = pd.read_excel("./data fixed/contatti.xlsx")

def extract_datetime(date_string):
    """
    Extract date from the contact date string
    """
    date = str(date_string)
    day = int(date[6:])
    month = int(date[4:6])
    year = int(date[:4])
    return pd.datetime(year, month, day)
    

def sparkasse_purchased(df, customer_id, contact_date, look_back = 31):
    """
    Returns whether the customer bought something after the contact date.
        Returns
        (1, most recent emission date) if present in the insurance purchase table(df)
        (0, 0) else if present in the insurance purchase table(df)
    """
    
    
    df_customer_purchase = df[(df["ndg_contraente"] == customer_id) & (df["emission date"] >= (contact_date - timedelta(days=look_back)))]
    
    if df_customer_purchase.shape[0] > 0:
        return (1, df_customer_purchase["emission date"].min())
    else:    
        return (0,0)
    
    
 
    
def get_purchase_label(df_assicurativo, df_contact, ndg_person):
    """
    Returns for that person the bought labels only
    """
    
    contact_date = df_contact[df_contact["NDG_cod"] == ndg_person]["contact date"].min()
    sp = sparkasse_purchased(df_assicurativo, ndg_person, contact_date)
   
    results = []
    if sp[0] > 0:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": sp[1],
                "type":"S",
                "purchase": 1})
   
         
    else:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":" ",
                "purchase": 0})
            
    return results
    
  
def get_purchase_targets(df_assicurativo, df_contact, contacted_persons):
    """
    For contacted people, find out the purchase labels for them
    """
    
    results = []
    for ndg_person in contacted_persons:
        results.append(get_purchase_label(df_assicurativo, df_contact, ndg_person))
        
    return results 


## Opening the insurances bought in each of the macro categories
categories = ["Property", "Malattia", "Infortunio", "RC tutela legale"]

for category in categories:

    # Contacts 2018
    df_contacts["contact date"] = df_contacts["Data del Contatto"].apply(extract_datetime)
    df_contacts_2018 = df_contacts[df_contacts["contact date"]  < pd.datetime(2019, 1, 1)]


    # Insurance traces
    df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                                  parse_dates = ["start date","end date","emission date"],
                                  sep =";")
     

    # Insurances sold only in 2018
    df_assicurativo_2018 = df_assicurativo[(pd.datetime(2018, 1, 1) <= df_assicurativo["emission date"]) 
                                           & (df_assicurativo["emission date"]  < pd.datetime(2019, 1, 1))]

    df_assicurativo_2018_prop = df_assicurativo_2018[df_assicurativo_2018["category"] == category]


    # Customers contacted in 2018 campaign
    customers_2018 = set(df_contacts_2018["NDG_cod"])
    len(customers_2018 & set(df_assicurativo_2018_prop["ndg_contraente"]))


    # Finding what was purchases made in each macro category for 2018 only 
    results = get_purchase_targets(df_assicurativo_2018_prop, df_contacts_2018 , customers_2018)

    df_targets_2018 = pd.DataFrame([i[0] for i in results])

    # Saving the purchases
    df_targets_2018.to_csv("./profile 2018/targets {}.csv".format(category), index = False, sep = ";")

