# -*- coding: utf-8 -*-
"""
Prepare insurance data by renaming old insurances in terms of the current/on the shelf
ones and opens up quadra pro family and Protection. This script handles only the breakdown till August 2019, not after that. 
Insurances after them are handled in different way as the data provided for break down of protection was slightly different.
@author: at80874
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

assicurativo_path = r"./assicurativo"

def process_dates(df):    
    """
    Process dataframes time fields
    """
    
    def extract_datetime(date_string):
        if pd.isna(date_string):
            date = pd.to_datetime("today")
            day = date.day
            month = date.month
            year = date.year
        
        else:        
            date = date_string.split("/")
            day = int(date[0])
            month = int(date[1])
            year = int(date[2].strip().split(" ")[0])
        return pd.datetime(year, month, day)
    
    df["emission date"] = df["data_emissione"].apply(lambda x: extract_datetime(x))
    df["start date"] = df["data_decorrenza"].apply(lambda x: extract_datetime(x))
    df["end date"] = df["data_scadenza"].apply(lambda x: extract_datetime(x))
    df["last payment date"] = df["data_ultimo_versamento"].apply(lambda x: extract_datetime(x))
    return df
    


def items_in_quadra_pro(df, ndg_code):
    """
    Items found in quadra pro for the item user bought
    """
    return list(df[df["NDG_cod"] == ndg_code]["new Garanzia"].values)



def open_quadra_pro(df):
    """
    Opens quadra pro and adds insurances in terms of on the shelf inside them
    """
    
    # Quadra Pro compositions
    df_pro_family = pd.read_excel(assicurativo_path + "/estrazione_dati_sparkasse_20190826_V1.xlsx", 
                                  sheet_name = "Pro Family")
    
    df_protection = pd.read_excel(assicurativo_path + "/estrazione_dati_sparkasse_20190826_V1.xlsx", 
                                  sheet_name = "Protection")
    
    df_pro_family_and_protection = pd.concat([df_pro_family, 
                                             df_protection])
    
    
    df_pro_family_and_protection_new = pd.read_excel(assicurativo_path + "/breakdown campagna 2019 con ndg codificato.xlsx", 
                                                     parse_date = "DT_EFFETTO")
    
    
    
    
    df_pro_family_and_protection_new = df_pro_family_and_protection_new[~df_pro_family_and_protection_new["ndg_codificato"].isna()]
    
    
    df_pro_family_and_protection_new["ndg_codificato"] = df_pro_family_and_protection_new["ndg_codificato"].astype("int64")
    
    
    
    df_new = pd.DataFrame()
    
    
    df_new["Garanzia"] = list(df_pro_family_and_protection_new["DESCRIZIONE_GARANZIA"]) + list(df_pro_family_and_protection["Garanzia"])
    
    
    df_new["NDG_cod"] = list(df_pro_family_and_protection_new["ndg_codificato"]) + list(df_pro_family_and_protection["NDG_cod"])
    
    
    
    df_mapping_new_old = pd.read_excel(assicurativo_path + "/quadra_pro_family_mappings.xlsx")
    
    new_old_mapping = dict(zip(df_mapping_new_old["New Quadra Pro"].apply(lambda x: x.lower()), 
                               df_mapping_new_old["Mapping old code"].values))
    
        
    def new_to_old_mapping(x):
        """
        Returns the mapping from new to old
        """
        return new_old_mapping[x.lower()]  
    
    
    df_pro_family_and_protection["new Garanzia"] = df_pro_family_and_protection["Garanzia"].apply(new_to_old_mapping)
    df_pro_family_and_protection = df_pro_family_and_protection[df_pro_family_and_protection["new Garanzia"].notnull()]
    
        
    result = []
    missing_quadra = []
    for index, row in df.iterrows():
        insurance_old = row.to_list()
        new_row = insurance_old   
        if row["cod_abi_3"] == "QUADRA PRO FAMILY" or row["cod_abi_3"] == "PROTECTION":
            ndg_code = row["ndg_contraente"]   
            contains = items_in_quadra_pro(df_pro_family_and_protection , ndg_code)
            if len(contains) == 0:
                print(row["cod_abi_3"], row["emission date"])
                missing_quadra.append(ndg_code)
                new_row = insurance_old + [row["cod_abi_3"]]
                result.append(new_row)
            for new_product in contains:
                print("No composition for ", row["cod_abi_3"], row["emission date"])
                new_row = insurance_old + [new_product]
                result.append(new_row)
        else:
            new_row = insurance_old + [row["cod_abi_3"]]
            result.append(new_row)
                              
            
    df_assicurativo_new = pd.DataFrame(result)
    df_assicurativo_new.columns = list(df_assicurativo_focus.columns) + ["Sub product"]
    df_assicurativo_new = df_assicurativo_new[df_assicurativo_new["Sub product"].notnull()]
    
    return df_assicurativo_new



def mappings_old_products_in_terms_new(df):
    """
    Maps old insurances in terms of current on the shelf products
    """
    
    
    df_mapping_old_new = pd.read_excel(assicurativo_path + "/old_new_mapping.xlsx",
                              sheet_name="Old to new Mapping")
    
    df_mapping_new_category = pd.read_excel(assicurativo_path + "/old_new_mapping.xlsx",
                              sheet_name="Structure new products",
                              header=1)
    
    df_mapping_new_category = df_mapping_new_category[df_mapping_new_category["Product"] == "PROTECTION"][["Cover","Code","Catergory"]]
    
    
    
    df_mapping_dictionary = df_mapping_old_new.merge(df_mapping_new_category , how = "left", on = "Code")
    
    
    def get_new_products_category(old_product, df_mapping):
        df_result = df_mapping[df_mapping["Old Products"] == old_product][["Cover_x","Catergory","Code"]]
        df_result.columns = ["Current Product", 
                             "Category",                            
                             "Code"]
        return [(rows["Current Product"], rows["Category"], rows["Code"]) for index, rows in df_result.iterrows()]
        
    
    
    product_mapping = dict()
    for old_product in set(df_mapping_dictionary["Old Products"]):
        result = get_new_products_category(old_product, df_mapping_dictionary)
        product_mapping[old_product] = result
    
    
    result = []
    for index,row in df.iterrows():
        insurance_old = row.to_list()
        new_row = insurance_old
        try:
            for new_product, category, code in product_mapping[row["Sub product"]]:
                new_row = insurance_old + [new_product, category, code]
                result.append(new_row)
        except:
            pass
            
    
    df_assicurativo_new = pd.DataFrame(result)
    df_assicurativo_new.columns = list(df.columns) + ["new product name","category","Code"]
    
    # Get the four major classes
    df_assicurativo_new = df_assicurativo_new[df_assicurativo_new["category"].isin(["Property","RC tutela legale","Infortunio","Malattia"])]
    
    
    
    def add_product_name_code(row):
        
        if row["new product name"] == "ACQUIRED INSURANCES":
            return  row["new product name"]
        
        elif row["new product name"] == "NO CURRENT MAPPING":
            return  row["new product name"]
        
        else:
            return row["new product name"] + "(" + str(row["Code"])+ ")"
        
        
        
    df_assicurativo_new["new product name code"] = df_assicurativo_new.apply(add_product_name_code, axis = 1)

    return df_assicurativo_new
    
if __name__ == "__main__":
        
    print("Processing Insurances")
    
    df_assicurativo = pd.read_csv(assicurativo_path + "/Unica_Assicurativo_v1.csv",
                                  sep =";")
    
    
    df_assicurativo = df_assicurativo.drop_duplicates()
    df_assicurativo = df_assicurativo[list(df_assicurativo.columns)[1:]]
    
    
    # Assicurativo focus
    df_important_assicurativo = pd.read_excel(assicurativo_path + "/Polizze.xlsx")
    
    # Target insurances
    insurances_focus = set(df_important_assicurativo["COD. ABI"]) 
    df_assicurativo_focus = df_assicurativo[df_assicurativo["cod_abi_3"].isin(insurances_focus)].copy()
    df_assicurativo_focus = process_dates(df_assicurativo_focus)
    
    
    print("Opening Quadra Pro and PROTECTION")
    # Fixing quadra PRO composition
    df_assicurativo_no_quadra = open_quadra_pro(df_assicurativo_focus)
    
    print("Naming old insurances in terms of the new on the shelf products")
    # Mapping old insurances in terms of current on the table
    df_assicurativo_new = mappings_old_products_in_terms_new(df_assicurativo_no_quadra)
    
    # Saving the new insurance trace file
    df_assicurativo_new.to_csv(assicurativo_path + "/Assicurativo_new_unica_v1.txt", 
                               sep = ";",
                               index = False)
    

