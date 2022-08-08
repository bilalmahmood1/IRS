# -*- coding: utf-8 -*-
"""
Prepare prediction targets, whether someone bought an insurance or not, for the customers contacted during the campaigns
@author: BM387
"""


import pandas as pd
from datetime import timedelta
from pandas import Timestamp



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
    
    
    
def alternate_purchased(df, customer_id):
    """
    Returns whether the customer bought something from another supplier.
        Returns
        (1, contact date) if purchase by the other supplier
        (0, 0) else
    """
    
    df_customer_purchase = df[(df["NDG_cod"] == customer_id) & 
                              (df["Descrizione sotto esito"] == ' Non interessato – Purchased throught Competitor')]
    
  
    if df_customer_purchase.shape[0] > 0:
        return (1, df_customer_purchase["contact date"].min())
    else:    
        return (0,0)
    
    
def get_purchase_label(df_assicurativo, df_contact, ndg_person):
    """
    Returns for that person the bought labels only
    """
    
    contact_date = df_contact[df_contact["NDG_cod"] == ndg_person]["contact date"].min()
    sp = sparkasse_purchased(df_assicurativo, ndg_person, contact_date)
    ap = alternate_purchased(df_contact, ndg_person)

    results = []
    if sp[0] > 0:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": sp[1],
                "type":"S",
                "purchase": 1})
    
    if ap[0]> 0:
         results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":"A",
                "purchase": 1})

         
    if sp[0] == 0 and ap[0] == 0:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":" ",
                "purchase": 0})
            
    return results
  
    
  
def get_purchase_label_test(df_assicurativo, df_contact, ndg_person):
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


def get_purchase_targets_test(df_assicurativo, df_contact, contacted_persons):
    """
    For contacted people, find out the purchase labels for them
    """
    
    results = []
    for ndg_person in contacted_persons:
        results.append(get_purchase_label_test(df_assicurativo, df_contact, ndg_person))
        
    return results 





    
def get_purchase_label_contact(df_contact, ndg_person):
    """
    Returns for that person the bought labels only
    """
    
    contact_date = df_contact[df_contact["NDG_cod"] == ndg_person]["contact date"].min()
    sp = sparkasse_purchased(df_assicurativo, ndg_person, contact_date)
    ap = alternate_purchased(df_contact, ndg_person)

    results = []
    if sp[0] > 0:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": sp[1],
                "type":"S",
                "purchase": 1})
    
    if ap[0]> 0:
         results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":"A",
                "purchase": 1})

         
    if sp[0] == 0 and ap[0] == 0:
        results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":" ",
                "purchase": 0})
            
    return results

def get_purchase_targets_contact(df_contact, contacted_persons):
    """
    For contacted people, find out the purchase labels for them
    """
    
       
    results = []
    for ndg_person in customers_2019:
        df_contact_ndg = df_contacts_2019[df_contacts_2019["NDG_cod"] == ndg_person]
           
        
        df_contact_ndg["Descrizione sotto esito"]
        
        df_contact_ndg["Descrizione sotto esito"].isin(["Vendita Properties","Vendita Personal",
                                          "Vendita Properties + Personal"])
        
        if df_contact_ndg["Descrizione sotto esito"].isin(["Vendita Properties","Vendita Personal",
                                          "Vendita Properties + Personal"]).sum() > 0:
            
          
            contact_date = df_contact_ndg["contact date"].min()
            
            results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":"S",
                "purchase": 1})
                      
        
        elif df_contact_ndg["Descrizione sotto esito"].isin([" Non interessato – Purchased throught Competitor"]).sum() > 0:
            
            contact_date = df_contact_ndg["contact date"].min()
            results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":"A",
                "purchase": 1})
            
        else:
          
            contact_date = df_contact_ndg["contact date"].min()
            results.append({"ndg_contraente":ndg_person,
                "contact date": contact_date,
                "purchase date": contact_date,
                "type":"",
                "purchase": 0})
        


    return results




# Contacts 2018

df_contacts = pd.read_excel("./data fixed/contatti.xlsx")
df_contacts["contact date"] = df_contacts["Data del Contatto"].apply(extract_datetime)
df_contacts_2018 = df_contacts[df_contacts["contact date"]  < pd.datetime(2019, 1, 1)]



df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                              parse_dates = ["start date","end date","emission date"],
                              sep =";")
 

df_assicurativo_2018 = df_assicurativo[(pd.datetime(2018, 1, 1) <= df_assicurativo["emission date"]) 
                                       & (df_assicurativo["emission date"]  < pd.datetime(2019, 1, 1))]
customers_2018 = set(df_contacts_2018["NDG_cod"])
len(customers_2018 & set(df_assicurativo_2018["ndg_contraente"]))
results = get_purchase_targets(df_assicurativo_2018, df_contacts_2018 , customers_2018)

df_targets_2018 = pd.DataFrame([i[0] for i in results])
df_targets_2018.to_csv("./profile 2018/targets.csv", index = False, sep = ";")



# Insurnaces 2019
df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                              parse_dates = ["start date", "end date", "emission date"],
                              sep =";")


# Insurances at the start of the campaign
df_assicurativo_2019 = df_assicurativo[(pd.datetime(2019, 8, 1) <= df_assicurativo["emission date"])]
len(set(df_assicurativo_2019["ndg_contraente"]))



df_contacts["contact date"] = df_contacts["Data del Contatto"].apply(extract_datetime)
df_contacts_2019 = df_contacts[df_contacts["contact date"]  >= pd.datetime(2019, 1, 1)]


customers_2019 = set(df_contacts_2019["NDG_cod"])
results = get_purchase_targets(df_assicurativo_2019, df_contacts_2019 , customers_2019)
df_targets_2019 = pd.DataFrame([i[0] for i in results])
df_targets_2019[df_targets_2019["type"] == "S"]

df_targets_2019.to_csv("./profile 2019/targets.csv", index = False, sep = ";")


# Reading test campign    
df_contacts_test = pd.read_csv("./data test/test_campaign.csv",sep = ";")
df_contacts_test["contact date"] = df_contacts_test["Data del Contatto"].apply(extract_datetime)


# Insurnaces sold
df_assicurativo = pd.read_csv("./assicurativo/Assicurativo_new_unica_v1.txt",
                              parse_dates = ["start date", "end date", "emission date"],
                              sep =";")



df_assicurativo_2019 = df_assicurativo[(pd.datetime(2019, 1, 1) <= df_assicurativo["emission date"]) 
                                       & (df_assicurativo["emission date"]  < (pd.datetime(2020, 1, 1)))]


customers_test = set(df_contacts_test["NDG_cod"])
len(customers_test & set(df_assicurativo_2019["ndg_contraente"]))
results = get_purchase_targets_test(df_assicurativo_2019, 
                                    df_contacts_test , 
                                    customers_test)

df_targets_test = pd.DataFrame([i[0] for i in results])
df_targets_test.to_csv("./profile test/targets.csv", index = False, sep = ";")