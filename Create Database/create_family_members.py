# -*- coding: utf-8 -*-
"""
Create the family member part of the customer profile which is done by counting how many members are there in the SCO relation.
This is done for test campaign, 2018 campaign and 2019 campaign.
@author: BM387
"""


from help_profiling_functions import *


years = ["test", 2018, 2019]

for year in years:

	## Get target customer list
	df_targets = pd.read_csv(r"./profile {}/targets.csv".format(year),
	                              sep = ";")

	# Count how many members there are in SCO relationship
	df_targets["members"] = df_targets["ndg_contraente"].apply(lambda x: len(mappings[get_sco_accounts(x, mappings)]) if get_sco_accounts(x, mappings) else 0)
	df_members_profile = df_targets[["ndg_contraente","members"]]


	cols = list(df_members_profile.columns)
	cols[0] = "ndg_codificato"
	df_members_profile.columns = cols


	# Saving the members profile
	df_members_profile.to_csv("./profile {}/members.csv".format(year), 
	                                   sep = ";",
	                                   index = False)





