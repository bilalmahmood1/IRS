# -*- coding: utf-8 -*-
"""
Set of all the helper functions used during the process of model building. Not all of them are used in the final analysis as different 
directions were used as the project evolved. 
@author: at80874
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from itertools import combinations
from tpot_config import classifier_config_dict


# Fea plotting settings
sns.set(style="white")
plt.style.use("default")


# Random seed for data splitting and pipeline search
seed = 123

# Number of cross validation folds to experiment
outer_folds = 5


def sigmoid(x):
    """
    Calculates the sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def binarize_features(df, boolean_columns):
    """
    Binarize all the columns in the df
    """
    for f in boolean_columns:
        df.loc[df[f] != 0, f] = 1  


def median_buckets(df, bucket_columns):
    """
    Make two levels for the columns provided, all the values above median
    gets the value of 1 and below gets 0
    """
    for c in bucket_columns:    
        df_c = df[c] > df[c].median()
        df[c] = df_c.apply(lambda x: 1 if x else 0)
        

def remove_features_list(l,r):
    """
    Remove the list of features in l give r and all the comune
    """
    for i in r:
        if i in l:    
            l.remove(i)
            
    return l

def remove_comune(l):
    """
    Remove the comune if present
    """
    find_comune = []
    
    for i in l:
        
        if "COMUNE" in i:
            find_comune.append(i)
            
    for i in find_comune:
         l.remove(i)
    
    return l


def create_ROC(y_test, y_pred,  title = "", label = "", path = ".", plot = True):
    """
    Providing the y_test and y_pred probabilities caluclate the AUC, and 
    FPR, TPR and Thresholds 
    """
    fpr, tpr, thresh = roc_curve(y_test, y_pred)
    area_under_curve = metrics.auc(fpr, tpr)
    
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve {} {}'.format(label, round(area_under_curve,2)))
        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()   
        plt.savefig(path, dpi = 300)
        
    return [area_under_curve,fpr, tpr, thresh]


def precision_recall(df, n):    
    """
    Precision Recall at n
    """

    total_rel = df["y_test"].sum()
    df_head = df.sort_values(by = "y_pred_score", 
                             ascending = False).head(n)

    df_prec = df_head["y_test"]
    rel = df_prec.sum()
    prec = df_prec.mean()   
    recall = rel / total_rel
    return prec, recall


def create_precision_recall_curve(df_prediction_test, steps, title, path = ".", plot = True):
    """
    Create precision recall number of recommendations
    """
     
    precisions = []
    recalls = []
    f1s = []
    step = 10
    num_recommedations = []
    total_number = df_prediction_test.shape[0]
    
    # To avoid perfect division with 10 because it leads to unaligned axis
    if total_number % 10 == 0:
        total_number += 1
        
    for i in range(step, total_number, step):
        p,r = precision_recall(df_prediction_test, i)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * ((p * r) / (p + r)))
        num_recommedations.append(i)


    df_pre_recall = pd.DataFrame({"Number of Customers Contacted": num_recommedations,
                                  "Precision Score": precisions,
                                  "Recall Score": recalls,
                                  "F1 Score": f1s})
    
    
    df_pre_recall = df_pre_recall.set_index("Number of Customers Contacted")
    
    if plot:    
        df_pre_recall.plot(title = title,  grid = True)
        plt.savefig(path, dpi = 300)
        
    return df_pre_recall
    
    
    

def get_counts_purchase(df, features):
    """
    Find out rate of purchase for the features selected
    """
    
    result = []
    for f in features:
        if 'Intercept' not in f:
            counts = df[df[f] > 0].shape[0]
            purchase_rate =  df[df[f] > 0]["purchase"].mean()
            
            negative_counts = df[df[f] == 0].shape[0]  
            negative_purchase_rate =  df[df[f] == 0]["purchase"].mean()
          
            if negative_purchase_rate == 0:
                print("Feature whose negative never got", f)
                boost = np.inf
                
            else:
                boost = purchase_rate / negative_purchase_rate
                
            result.append([f, counts, purchase_rate, negative_counts, negative_purchase_rate, boost])
        else:
            result.append([f, 0, 0, 0, 0,0])
    
    return pd.DataFrame(result, columns = ["Features", "Positive Contacts", "Purchases Rate",
                                           "Negative Contacts", "Negative Purchases Rate", "Boost"])


def number_of_purchases(row):
    """
    Finding number of payments for each insurance purchase which are 
    then expanded when adding renewals
    """
    last_payment = row["last payment"]
    start_date = row["start date"]
    end_date = row["end date"]
    payments = (last_payment - start_date).days // (end_date - start_date).days
    
    remainder = (last_payment - start_date).days % (end_date - start_date).days
    if payments == 0:
        return 1
    if remainder > 0:
        return payments  + 1
    else:
        return payments
    
def extract_datetime(date_string):
    """
    Extracts date from the string
    """
    date = date_string.split("/")
    day = int(date[0])
    month = int(date[1])
    year = int(date[2].strip().split(" ")[0])
    return pd.datetime(year, month, day)
    


def feature_importance_plot(df_result,
                            n = 50, 
                            ascending = False, 
                            title = "",
                            path = ".",
                            plot = True):
    
    """
    Produces feature score plots
    """ 
    
    df_feature_scores = df_result.sort_values(by="score", ascending=ascending)
    df_feature_scores_n = df_feature_scores.iloc[0:n,:]
    
    if plot:
        df_feature_scores_n.plot(x="Features",
                                 y="score",
                                 color = "black",
                                 kind ="bar",
                                 legend = None)                            
    
        
        plt.title(title)
        plt.ylabel("Coeff")
        plt.xlabel("User profile features")
        plt.tight_layout()
        plt.savefig(path,
                    bbox_inches = "tight")
    
        
       
        
def PRF1_plot_CV(df_result,
                 title = "",
                 path = "."):
    
    """
    Produces CV PR curves
    """ 
    
    df_result.plot(title = title)
    plt.ylabel("Score")
    plt.grid("on")
    plt.savefig(path,
                bbox_inches = "tight")
    
    
    
def feature_importance_plot_CV(df_result, whole_data, 
                            n = 50, 
                            ascending = False, 
                            title = "",
                            path = ".",
                            plot = True):
    
    """
    Produces feature score plots
    """ 
    
    df_feature_scores = df_result.sort_values(by="Average score", ascending=ascending)
    df_feature_scores_n = df_feature_scores.iloc[0:n,:]
    
    if plot:
        df_feature_scores_n.plot(x="Features",
                                 y="Average score",
                                 color = "black",
                                 kind ="bar",
                                 legend = None)                            
    
        
        plt.title(title)
        plt.ylabel("Coeff")
        plt.xlabel("User profile features")
        plt.tight_layout()
        plt.savefig(path,
                    bbox_inches = "tight")
    

def measure_performance_n(df_prediction_test, n):
    """
    Measures performance at the specified prediction level
    """
        
    precision_at_n = df_prediction_test.sort_values(by = "y_pred_score", ascending = False).head(n)["y_test"].mean()
    
    recall_at_n = df_prediction_test.sort_values(by = "y_pred_score", ascending = False).head(n)["y_test"].sum()/ df_prediction_test.sort_values(by = "y_pred_score", ascending = False)["y_test"].sum()
    
    f1_at_n = 2 * (precision_at_n * recall_at_n) / (precision_at_n + recall_at_n)

    return precision_at_n, recall_at_n, f1_at_n


def measure_statistics(y_train, y_train_pred, y_test, y_test_pred):
    """
    Calculates performance statistics for train and test sets
	"""
    results = {"Train": { "accuracy": accuracy_score(y_train, y_train_pred),
                          "precision": precision_score(y_train, y_train_pred),
                          "recall": recall_score(y_train, y_train_pred),
                          "f1-score": f1_score(y_train, y_train_pred ) }
                          , 
                "Test": { "accuracy": accuracy_score(y_test, y_test_pred),
                          "precision": precision_score(y_test, y_test_pred),
                          "recall": recall_score(y_test, y_test_pred),
                          "f1-score": f1_score(y_test, y_test_pred),
                          "Total recommended": y_test_pred.sum(),
                          "Number of customers": y_test_pred.shape[0]}}
    return results

def process_cv_performance(results_ps, experiment):
    """
    Calculates average of AUC for the folds
    """
    
    cross_validated_auc = [i["Test"]["AUC"] for i in results_ps]
    avg_auc = np.mean(cross_validated_auc)
    std_auc = np.std(cross_validated_auc)

    return avg_auc, std_auc



def process_cv_feature_scores(results_f, experiment):
    """
    Calculates average of feature score importance for the folds
    """
    
    ## Experiment
    print("Experiment: ", experiment)

    ## Cross validation scores
    print("Cross validation scores")
    print(results_f)

    result_array = []
        
    for df_f in results_f:
        f_array = df_f["score"].values.reshape(-1,1)
        result_array.append(f_array)
        
  
    result_array = np.array(result_array)    
    mean_scores = result_array.mean(axis = 0).ravel()
    
    return  pd.DataFrame({"Features": results_f[0]["Features"].values, "Average score": mean_scores})
   

    
def process_cv_pr_curves(results_pr, experiment):
    """
    Calculates average of feature score importance for the folds
    """
    
    result_array_p = []
    result_array_r = []
    result_array_f1 = []

       
    for df_f in results_pr:
        f_array_p = df_f["Precision Score"].values.reshape(-1,1)
        result_array_p.append(f_array_p)
        
        f_array_r = df_f["Recall Score"].values.reshape(-1,1)
        result_array_r.append(f_array_r)
        
        
        f_array_f1 = df_f["F1 Score"].values.reshape(-1,1)
        result_array_f1.append(f_array_f1)
        
        
    result_array_p = np.array(result_array_p)    
    result_array_r = np.array(result_array_r)    
    result_array_f1 = np.array(result_array_f1)    
    
    mean_scores_p = result_array_p.mean(axis = 0).ravel()
    mean_scores_r = result_array_r.mean(axis = 0).ravel()
    mean_scores_f1 = result_array_f1.mean(axis = 0).ravel()
    
    df_pr_cv_mean = pd.DataFrame({"Number of Customers Contacted": results_pr[0].index,
                          "Average Precision score": mean_scores_p,
                          "Average Recall score": mean_scores_r,
                          "Average F1 score": mean_scores_f1,
                          })
    
    df_pr_cv_mean = df_pr_cv_mean.set_index("Number of Customers Contacted")
    return  df_pr_cv_mean




def build_evaluated_model(X_train, X_test, y_train, y_test, classifier_config_dict,
                          experiment, fold,
                          exp_path,
                          cols_use,
                          top_features = 25,
                          plot = False):
    """
    Builds the best pipeline using TPOT, saves the best model,
    measures ROC, PR curve, basic stats.    
    """
    
    
    ## Best Model building using cross-validation of the inner loop, the models to search are limited to Logistic regression using the self defined config_dict                                     
    tpot = TPOTClassifier(generations = 50, population_size = 50,
                          early_stop = 10, verbosity=2, 
                          cv=5,
                          template='Transformer-Classifier',
                          config_dict = classifier_config_dict,
                          scoring= "roc_auc",           
                          n_jobs = -1,
                          random_state=seed)
    
    
    
    print("Fitting model")
    # Fitting the best model
    tpot.fit(X_train, y_train) 
    # Saving the best model
    tpot.export(exp_path + '/tpot_best_model for fold {} of Experiment {}.py'.format(fold, experiment))
    
    
    print("Evaluating model") 
    ## Model Evaluation    
    y_pred_train = tpot.predict(X_train)
    y_pred_test = tpot.predict(X_test)

    y_pred_train_pa = tpot.predict_proba(X_train)[:,1]
    y_pred_test_pa = tpot.predict_proba(X_test)[:,1]


    auc_test, fpr, tpr, thresh = create_ROC(y_test, y_pred_test_pa, label = "Fold {}".format(fold),
               title = "ROC curve for fold {} of EXPERIMENT {}".format(fold, experiment),
               path = exp_path +  "/Test ROC Experiment {} fold {}.jpeg".format(experiment, fold),
               plot = plot)
    

    roc = (fpr, tpr, thresh)
    
    performance_statistics = measure_statistics(y_train, y_pred_train, y_test, y_pred_test)
    performance_statistics["Test"]["AUC"] = auc_test    
    
    
    df_prediction_test = pd.DataFrame({"y_test": y_test, "y_pred_score": y_pred_test_pa})
    df_prediction_test.sort_values("y_pred_score", ascending = False)
    df_pre_recall = create_precision_recall_curve(df_prediction_test, 100, 
                    title = "Precision Recall Curve  for fold {} of EXPERIMENT {}".format(fold, 
                                                                                          experiment), 
                                                  path = exp_path +  "/Test Precision Recall Curve Experiment {} fold {}.jpeg".format(experiment, fold),
                                                  plot = plot)
    
    
    df_features = None

    """
    best_pipeline = tpot.fitted_pipeline_   
    best_clf = best_pipeline["logisticregression"]
   
    
    df_features = pd.DataFrame([cols_use, 
                          best_clf.coef_[0]]).T
    
   
    df_features.columns = ["Features","score"]
   
    df_intercept = pd.DataFrame([["Intercept", best_clf.intercept_[0]]],
                                columns=df_features.columns)

    df_features = df_features.append(df_intercept)
   
    # Produces features importance plots
    feature_importance_plot(df_features,
                   n = top_features,
                   title = "Top features({}) for fold {} of Experiment {}".format(top_features, fold, experiment),
                   path = exp_path +  "/Top model features for Experiment {} fold {}.jpeg".format(experiment, fold),
                   plot = plot)
   
  
    feature_importance_plot(df_features,
                   n = top_features,
                   ascending = True,
                   title = "Top features({}) for fold {} of Experiment {}".format(top_features, fold, experiment),
                   path = exp_path +  "/Bottom model features for Experiment {} fold {}.jpeg".format(experiment, fold),
                   plot = plot)
   
    """

    return tpot, performance_statistics, roc, df_pre_recall, df_features

 
def run_experiment(X_data, y_data, cols_use, experiment, exp_path, desc, whole_data, category):
    """
    Runs different experiments on the data inputed. Tunes the best model for the chosen experiment and measures the cross validated performance  
    """

    # Number of top/Bottom features to select
    top_feature = 20
    
    print("Running experiment {}".format(experiment))
    print("Data, shape", X_data.shape)


    # Fold results
    results_pipe = []
    results_ps = []
    results_roc = []
    results_pr = []
    results_f = []
    
    # Outer folds
    outer_cv =  StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state = seed)
    
    
    fold = 0
    for train_index, test_index in outer_cv.split(X_data, y_data):
        fold = fold + 1
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
    
        print("Fold", fold)
        print("Training data shape: ", X_train.shape)
        print("Training data purchasers: ", y_train.sum())
        print("Training data purchase rate: ", y_train.mean())
        
        
        print("Testing data shape: ", X_test.shape)
        print("Testing data purchasers: ", y_test.sum())
        print("Testing data purchase rate: ", y_test.mean())
        
        print("")
    
    	#Building the best model
        tpot, performance_statistics, roc, df_pre_recall, df_features = build_evaluated_model(X_train, X_test, y_train, y_test, 
                                                                                              classifier_config_dict,experiment, fold,
                                                                                              exp_path,cols_use, top_feature)
        
        # Saving the results
        results_pipe.append(tpot)
        results_ps.append(performance_statistics)

        print("Performance statistics: ", results_ps)
        results_roc.append(roc)
        results_pr.append(df_pre_recall)
        results_f.append(df_features)
        
    # Calculates average AUC score
    cv_performance = process_cv_performance(results_ps, experiment)
    
    
    print("Description experiment {}".format(experiment), desc)
    print("Score for experiment {}".format(experiment), cv_performance)
     

    cv_features = None
    """
    # Calculates average cross-validation score
    cv_features = process_cv_feature_scores(results_f, experiment)
    
    
    # Calculates cross validated features importance plots
    feature_importance_plot_CV(cv_features, whole_data,
                                n = top_feature,
                                title = "Top features({}) for Experiment {}".format(top_feature, experiment),
                                path = exp_path +  "/Top model features for Experiment {}.jpeg".format(experiment))
    
    feature_importance_plot_CV(cv_features, whole_data,
                               n = top_feature,
                               ascending = True,
                               title = "Bottom features({}) for Experiment {}".format(top_feature, experiment),
                               path = exp_path +  "/Bottom model features for Experiment {}.jpeg".format(experiment))
    
   """
    columns_counts = ["purchase"] +  cols_use
    
    # Calculates cross-validated precision and recall
    cv_pre_rec = process_cv_pr_curves(results_pr, experiment)
    print(cv_pre_rec)
    PRF1_plot_CV(cv_pre_rec,
                 "Cross Validated Precision Recall Curves for {}({}) with AUC: {} std: +/-{}".format(category, desc,
                                                                                                round(cv_performance[0], 3),
                                                                                                round(cv_performance[1], 3),
                                                                                                
                                                                                                ),
                 path = exp_path + "/Precision Recall Curve.jpeg")
        
    print("Done Running experiment {}".format(experiment))    


    return cv_performance, cv_pre_rec, cv_features


def make_description(data, branch = False): 
    """
    Make description of customer
    """ 
    all_features = list(data.columns) 
    id_columns = [i for i in all_features if "ndg_codificato" in i]
    
    
    gender_columns = [i for i in all_features if "(GENDER)" in i]
    
    age_columns = [i for i in all_features if "(AGE)" in i]
    cae_columns = [i for i in all_features if "(CAE)" in i]
    branch_columns = [i for i in all_features if "(BRANCH)" in i] 
    members_columns = ["members"]
    previous_purhcase_columns = ["previous purchase"]
    foriegner_columns = ["foreigner"]
    individual_money_columns = ["individual(debit + stock)", "individual salary", "individual wealth decrease","individual wealth increase"]
    family_money_columns = ["family(debit + stock)", "family salary", "family wealth decrease", "family wealth increase"]
    individual_spending_money_columns=  [i for i in all_features if "(POS)" in i and "individual" in i]
    family_spending_money_columns = [i for i in all_features if "(POS)" in i and "family" in i]
    
    

    descriptions = []    
    if branch:
            
        descr = """
                  <p> <b>Demographics: </b> </p>
                      <p> NDG: {} </p>
                      <p> Gender: {} </p> 
                      <p> Age: {} </p>
                      <p> Profession: {} </p>
                      <p> Members: {} </p>
                      <p> Foriegner: {} </p>
                      <p> Previous Purchase: {} </p>
                      <p> Branch : {} </p>
                  <p> <b>Finances: </b> </p>
                      <p> {} </p>
                      <p> {} </p>
                  <p> <b>Spending: </b> </p>
                      <p> {} </p>
                      <p> {} </p>
                """
                   
        for i in data.iterrows():
          
            ndg = str(i[1].loc[id_columns][0])
            gender = i[1].loc[gender_columns][0]
            
            age = i[1].loc[age_columns].nlargest(n=1).index[0]
            cae = i[1].loc[cae_columns].nlargest(n=1).index[0]
            branch = i[1].loc[branch_columns].nlargest(n=1).index[0]
            members = i[1].loc[members_columns][0]
            previous_purchase = i[1].loc[previous_purhcase_columns][0]
            foriegner = i[1].loc[foriegner_columns][0]
            individual_money= i[1].loc[individual_money_columns].to_string()
            family_money = i[1].loc[family_money_columns].to_string()
            
            individual_spending = i[1].loc[individual_spending_money_columns].nlargest(n=3).to_string()
            family_spending = i[1].loc[family_spending_money_columns].nlargest(n=3).to_string()
                 
            descriptions.append(descr.format(ndg, gender, age, cae, members, foriegner, 
                                            previous_purchase,branch, individual_money, family_money, 
                                            individual_spending, family_spending))
            
        return descriptions

    else:
        descr = """
                  <p> <b>Demographics: </b> </p>
                      <p> NDG: {} </p>
                      <p> Gender: {} </p> 
                      <p> Age: {} </p>
                      <p> Profession: {} </p>
                      <p> Members: {} </p>
                      <p> Foriegner: {} </p>
                      <p> Previous Purchase: {} </p>
                  <p> <b>Finances: </b> </p>
                      <p> {} </p>
                      <p> {} </p>
                  <p> <b>Spending: </b> </p>
                      <p> {} </p>
                      <p> {} </p>
                """
                   
        for i in data.iterrows():
          
            ndg = str(i[1].loc[id_columns][0])
            gender = i[1].loc[gender_columns][0]
            
            age = i[1].loc[age_columns].nlargest(n=1).index[0]
            cae = i[1].loc[cae_columns].nlargest(n=1).index[0]
            members = i[1].loc[members_columns][0]
            previous_purchase = i[1].loc[previous_purhcase_columns][0]
            foriegner = i[1].loc[foriegner_columns][0]
            individual_money= i[1].loc[individual_money_columns].to_string()
            family_money = i[1].loc[family_money_columns].to_string()
            
            individual_spending = i[1].loc[individual_spending_money_columns].nlargest(n=3).to_string()
            family_spending = i[1].loc[family_spending_money_columns].nlargest(n=3).to_string()
                 
            descriptions.append(descr.format(ndg, gender, age, cae, members, foriegner, 
                                            previous_purchase, individual_money, family_money, 
                                            individual_spending, family_spending))
            
        return descriptions
    


def make_co_occurance_portfolio_matrix(df_user_items_bought, plot = True, 
                                       figsize = (10,10), 
                                       title = "",
                                       threshold = 0):
    
    
    """
    Creates co_occurance portfolio matrix from the purchase history of insurances
    """
    
    # Ordering the insurances by how much they were sold     
    insurance_names = list(df_user_items_bought.sum().sort_values(ascending = False).index)
    co_occurance_matrix = pd.DataFrame(0, index= insurance_names, columns=insurance_names)
    for user_items in df_user_items_bought.iterrows():
        items_bought_vector = user_items[1]
        items_bought = list(items_bought_vector[items_bought_vector > 0].index)
        if len(items_bought) > 1:       
            for i, j in list(combinations(items_bought, r = 2)):
                co_occurance_matrix.loc[i,j] += 1
                co_occurance_matrix.loc[j,i] += 1
    if plot:
        filtered_matrix = co_occurance_matrix[co_occurance_matrix >= threshold] 
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize = figsize)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(filtered_matrix, annot=True, fmt = "d", cmap = "seismic", center = 0,
                    square=True, linewidths=.1, cbar = False)

        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.title(title)
        

    return co_occurance_matrix




def plot_features_boost_count(cv_features, whole_data, all_features, topn = 10, ascending = True, save_path = "."):
    """
    Plot Logistic regression model features along with PowerBI like Key Influencers
    """
    df_counts = get_counts_purchase(whole_data, all_features)
    
    df_res_sorted = cv_features.merge(df_counts, on = "Features").sort_values(by = "Average score", 
                                                                              ascending = ascending)

    df_res_sorted = df_res_sorted.set_index("Features")
    
    df_res_sorted = df_res_sorted[df_res_sorted["Positive Contacts"] > 20]
    
    plt.figure()
    
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.subplot(1,3,1)
    df_res_sorted[["Average score"]].head(topn).plot(kind = "bar", ax=plt.gca())
    
    ax=plt.gca()
    plt.legend(["Logistic Regresssion Coeffs"])
    plt.legend(["LR Coeffs"])
    
    plt.subplot(1,3,2)
    df_res_sorted[["Boost"]].head(topn).plot(kind = "bar", ax=plt.gca())
    
    
    ax=plt.gca()
    plt.legend(["x times purchase is more likely when this feature is not 0"])
    plt.legend(["x Times purchase rate"])
   
    plt.axhline(y=1.0, color='r', linestyle='-.')
   
    
    plt.subplot(1,3,3)
    df_res_sorted[["Positive Contacts"]].head(topn).plot(kind = "bar", ax=plt.gca())
    ax=plt.gca()
    plt.legend(["Number of customers with non zero value"])
    plt.legend(["Customers"])
    
    plt.axhline(y=20, color='r', linestyle='-.') 
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig(save_path, bbox_inches = "tight",dpi=300)
    plt.show()
    
    
  
def log_transform(df, columns):
    """
    Simply do the log transform of the columns added
    """
    
    for c in columns:
        df[c] = np.log10(np.abs(df[c]) + 1 )