# -*- coding: utf-8 -*-
"""
This is the main script that trains user-user nearest neighbor based classifier, 
produces their cross validated performance scores(ROC curve, Precision-Recall curves). 

@author: BM387
"""

# Importing the essential functions
import logging
from helper_functions_model import *
from datetime import datetime
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')

if __name__ == '__main__':

    logging.info("Starting the experiments")
    logging.info(datetime.now())
    
    # Target categories to build logistic regression models 
    categories = ["purchase","Property", "Malattia", "Infortunio", "RC tutela legale"]
    results = {}
    for category in categories:

        logging.info(category)
        
        # Reading profiled customers of 2018 campaign
        df_customer_profile_2018 = pd.read_csv("../Create Database/profile 2018/customer profile.csv".format(category), 
                                               sep = ";")
        
        
        #df_customer_profile_2018 = df_customer_profile_2018.head(500)
    
        # Focusing only on customers who bought insurance from Sparkasse or were not interested, excluding the customers who said they had insurance from 
        # alternative suppliers, type = "A"
        df_customer_profile_2018_S = df_customer_profile_2018[df_customer_profile_2018["type"].isin([" ","S"])]
    
    
    
        # Reading profiled customers of 2019 campaign
        df_customer_profile_2019 = pd.read_csv("../Create Database/profile 2019/customer profile.csv", 
                                                sep = ";")

    
        #df_customer_profile_2019 = df_customer_profile_2019.head(500)
        # Focusing only on customers who bought insurance from Sparkasse or were not interested, excluding the customers who said they had insurance from 
        # alternative suppliers, type = "A"
        df_customer_profile_2019_S = df_customer_profile_2019[df_customer_profile_2019["type"].isin([" ","S"])]
    
    
    
        # Reading profiled customers of 2018 + 2019 campaigns
        df_customer_profile_combined = pd.read_csv("../Create Database/profile 2018+2019/customer profile.csv", 
                                              sep = ";")
    
    
        #df_customer_profile_combined = df_customer_profile_combined.head(500)

        # Focusing only on customers who bought insurance from Sparkasse or were not interested, excluding the customers who said they had insurance from 
        # alternative suppliers, type = "A"
        df_customer_profile_combined_S = df_customer_profile_combined[df_customer_profile_combined["type"].isin([" ","S"])]
    
    
        # All customer features 
        all_features = df_customer_profile_2018.iloc[:, 15:].columns

    
        # Branches features
        branches = [i for i in df_customer_profile_2018.columns if "BRANCH" in i]
    
        # Money Columns
        money_columns = all_features[58:148]
    
        # Log transforming all the money columns 
        log_transform(df_customer_profile_2018_S, money_columns)
        log_transform(df_customer_profile_2019_S, money_columns)
        log_transform(df_customer_profile_combined_S, money_columns)
    

     
        # Defining 3 experiements: Training the model on 2018 campaign (Exp: 1), 2019 campaign(Exp: 2), and 2018 + 2019 campaign((Exp: 3)) 
        experiments = [          
            (1, 
             "2018 with branches with S",
             df_customer_profile_2018_S[all_features].values,
             df_customer_profile_2018_S[category].values,
             all_features, 
             df_customer_profile_2018_S)
            ,
            (2, 
             "2019 with branches with S",
             df_customer_profile_2019_S[all_features].values,
             df_customer_profile_2019_S[category].values,
             all_features, 
             df_customer_profile_2019_S)
            ,
            (3, 
             "2018 + 2019 with branches with S",
             df_customer_profile_combined_S[all_features].values,
             df_customer_profile_combined_S[category].values,
             all_features, 
             df_customer_profile_combined_S)]
           
    
        # Main experiment loop that trains the model, produces cross validated performance statistics, since we are focusing on logistic regression model,
        # it also produces the top and bottom cross validated featues
        exp_results = []
        for i in experiments:
          
            # Describing the data
            desc = i[1]
            logging.info(desc)
            
            # Reading the experiment data
            whole_data = i[5]
            
            
            logging.info("Experiment: ")
            logging.info(i[0])
            logging.info(datetime.now())

            experiment = i[0]
    
            # Experiment X features
            X_data  = i[2]
            # Experiment y features
            y_data = i[3]
            
            # Printing data shape
            logging.info("Data Shape")
            logging.info(X_data.shape)
            logging.info(y_data.shape)
            
            # Features to use
            cols_use = i[4]
    
            # Brief experiment description
           
            
            exp_path = ""
            # Path to store the results of the experiment
            exp_path = "./model {}/experiment {}".format(category, experiment)
    


            # Run the model training and validation step
            cv_performance,cv_pre_rec, cv_features = run_experiment(X_data, y_data, cols_use,
                                                                    experiment, exp_path, desc,
                                                                    whole_data)
            
    
            # Print Model performance statistics
            # Saving the results of each experiment
            exp_results.append((desc, cv_performance))


            logging.info("Results: ")
        
            logging.info("CV")
            logging.info(cv_performance)
            

            logging.info("Rank")
            logging.info(cv_pre_rec[["Average Precision score","Average Recall score"]].head(20))
    
