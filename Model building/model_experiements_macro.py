# -*- coding: utf-8 -*-
"""
This is the main script that trains 5 best tuned logistic regression models, produces their cross validated performance scores(ROC curve, Precision-Recall curves) 
along with their top and bottom features. The script stores all the results in separate folder(model _ ) along with the best models that are then used  
to set logistic regression models in KNIME for prediction.

@author: BM387
"""

# Importing the essential functions
from helper_functions_model import *

if __name__ == '__main__':

    print("XXXXXXXXXXXXXXXXXXXXXXXX")
    print("Starting the experiments")
    
    # Target categories to build logistic regression models 
    categories = ["purchase","Property", "Malattia", "Infortunio", "RC tutela legale"]
    
    for category in categories:
    
    
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
    

        """
            # Money Columns
            money_columns = all_features[58:148]
        
            print(money_columns)
            print(money_columns)
            # Log transforming all the money columns 
            log_transform(df_customer_profile_2018_S, money_columns)
            log_transform(df_customer_profile_2019_S, money_columns)
            log_transform(df_customer_profile_combined_S, money_columns)
        
        """
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
             df_customer_profile_combined_S)
            ]
           
    
        # Main experiment loop that trains the model, produces cross validated performance statistics, since we are focusing on logistic regression model,
        # it also produces the top and bottom cross validated featues
        exp_results = []
        for i in experiments:
          
            # Reading the experiment data
            whole_data = i[5]
                  
            print("Experiment:", i[0])
            experiment = i[0]
    
            # Experiment X features
            X_data  = i[2]
            # Experiment y features
            y_data = i[3]
            
            # Printing data shape
            print(X_data.shape)
            print(y_data.shape)
            
            # Features to use
            cols_use = i[4]
    
            # Brief experiment description
            desc = i[1]
            print(desc)
            
            exp_path = ""
            # Path to store the results of the experiment
            exp_path = "./model {}/experiment {}".format(category, experiment)
    
            # Run the model training and validation step
            cv_performance,cv_pre_rec, cv_features = run_experiment(X_data, y_data, cols_use,
                                                                    experiment, exp_path, desc,
                                                                    whole_data,
                                                                    category)
            
    
            # Print Model performance statistics
            print(category, ": AUC", cv_performance)
            print("PR@100", cv_pre_rec[["Average Precision score","Average Recall score"]].head(20))
            
            # Saving the results of each experiment
            exp_results.append((desc, cv_performance))
    
        print("Print top results from each of the experiment sets")
        print(exp_results)


    """
            # Saving the top features
            save_path = "./model {}/experiment {}/Bottom Features_Boost_Score_{}.jpg".format(category, 
                                                                          experiment, 
                                                                          category)
    
            plot_features_boost_count(cv_features, whole_data, all_features, topn = 10,
                                      ascending = True, save_path = save_path)
    
            # Saving the bottom features
            save_path = "./model {}/experiment {}/Top Features_Boost_Score_{}.jpg".format(category, 
                                                                          experiment, 
                                                                          category)
    
            plot_features_boost_count(cv_features, whole_data, all_features, topn = 10,
                                      ascending = False, save_path = save_path)
    
    """
