# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)
library(mcmcse)
library(dplyr)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="KailiHB",
  modelDescr ="Mixed logit model on Kaili ownership choice databasea, uncorrelated Lognormals in utility space - HB",
  indivID   = "X1",  
  HB         = TRUE,
  nCores    = 4
)

# ################################################################# #
#### LOAD databaseA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #
database = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/xmatswave2019_long.csv',header = TRUE)

# Scale one time costs and monthly mileage to $1000/1000 mi
database[,57] = database[,57] / 1000
database[,58] = database[,58] / 1000
database[,60] = database[,60] / 1000
database[,74] =  database[,74] / 1000
database[,89] = database[,89] / 1000
database[,91] = database[,91] / 1000

# Scale monthly costs to $100
database[,77] = database[,77] / 100
database[,78] = database[,78] / 100
database[,80] = database[,80] / 100
database[,84] = database[,84] / 100


# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(asc_pa=0, asc_as=0, asc_pca=0,
                av_cost=0, pc_dr_cost=0, pc_mo_cost=0,
                av_app_fee=0, av_mo_fee=0, av_dr_rate=0,
                pa_hhs=0, as_hhs=0, pca_hhs=0,
                pa_disab=0, as_disab=0, pca_disab=0,
                pa_veh_own=0, as_veh_own=0, pca_veh_own=0,
                pa_age=0, as_age=0, pca_age=0,
                pa_male=0, as_male=0, pca_male=0,
                pa_ft=0, as_ft=0, pca_ft=0,
                pa_pt=0, as_pt=0, pca_pt=0,
                pa_lic=0, as_lic=0, pca_lic=0,
                pa_bike=0, as_bike=0, pca_bike=0,
                pa_car=0, as_car=0, pca_car=0,
                pa_smart=0, as_smart=0, pca_smart=0,
                pc_lev1=0, pa_lev3=0, as_lev3=0,
                pa_lev4=0, as_lev4=0, pa_lev5=0,
                as_lev5=0, as_mile16=0, pca_mile16=0,
                as_mile44=0, pca_mile44=0, as_mileunl=0, 
                pca_mileunl=0
)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### HB settings                                                 ####
# ################################################################# #
# Setup cv matrix to have correlations between ASC and none for other variables
cvMat = diag(x=5, nrow=length(apollo_beta),ncol=length(apollo_beta))
cvMat[1,2] = 5
cvMat[2,1] = 5
cvMat[1,3] = 5
cvMat[3,1] = 5
cvMat[2,3] = 5
cvMat[3,2] = 5

apollo_HB = list(
  hbDist      = c(asc_pa="N", asc_as="N", asc_pca="N",
                  av_cost="N", pc_dr_cost="N", pc_mo_cost="N",
				  av_app_fee="N", av_mo_fee="N", av_dr_rate="N",
				  pa_hhs="N", as_hhs="N", pca_hhs="N",
				  pa_disab="N", as_disab="N", pca_disab="N",
				  pa_veh_own="N", as_veh_own="N", pca_veh_own="N",
				  pa_age="N", as_age="N", pca_age="N",
				  pa_male="N", as_male="N", pca_male="N",
				  pa_ft="N", as_ft="N", pca_ft="N",
				  pa_pt="N", as_pt="N", pca_pt="N",
				  pa_lic="N", as_lic="N", pca_lic="N",
				  pa_bike="N", as_bike="N", pca_bike="N",
				  pa_car="N", as_car="N", pca_car="N",
				  pa_smart="N", as_smart="N", pca_smart="N",
				  pc_lev1="N", pa_lev3="N", as_lev3="N",
				  pa_lev4="N", as_lev4="N", pa_lev5="N",
				  as_lev5="N", as_mile16="N", pca_mile16="N",
				  as_mile44="N", pca_mile44="N", as_mileunl="N", 
				  pca_mileunl="N"
				  ),
  gNCREP      = 2000, 
  gNEREP      = 2000, 
  gINFOSKIP   = 250,
  gFULLCV     = FALSE,
  hIW         = TRUE,
  pvMatrix    = cvMat, # Use same starting prior as Stan, N(0,5) and correlated ASC
  nodiagnostics = TRUE
)

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### ANALYSIS OF CHOICES                                         ####
# ################################################################# #
choiceAnalysis_settings <- list(
    alternatives  = c(pc=1, pa=2, as=3, pca=4),
    avail        = list(pc=1, pa=1, as=1, pca=1),
    choiceVar    = database[,50]
)
apollo_choiceAnalysis(choiceAnalysis_settings, apollo_inputs)
# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #
apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  ### Create list of probabilities P
  P = list()
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[['pc']] = database[,57]*av_cost + database[,67]*pc_dr_cost + database[,77]*pc_mo_cost + (database[,52]==1)*pc_lev1;
  V[['pa']] = asc_pa + database[,58]*av_cost + database[,6]*pa_hhs + database[,13]*pa_disab + database[,18]*pa_veh_own + (database[,13]<40)*pa_age + (database[,15]==1)*pa_male + (database[,20]==1)*pa_ft + (database[,20]==2)*pa_pt + database[,17]*pa_lic + database[,31]*pa_bike + database[,30]*pa_car + database[,32]*pa_smart + (database[,53]==3)*pa_lev3 + (database[,53]==4)*pa_lev4 + (database[,53]==5)*pa_lev5;
  V[['as']] = asc_as +                           + database[,74]*av_app_fee + database[,84]*av_mo_fee + database[,64]*av_dr_rate + database[,6]*as_hhs + database[,13]*as_disab + database[,18]*as_veh_own + (database[,13]<40)*as_age + (database[,15]==1)*as_male + (database[,20]==1)*as_ft + (database[,20]==2)*as_pt + database[,17]*as_lic + database[,31]*as_bike + database[,30]*as_car + database[,32]*as_smart + (database[,54]==3)*as_lev3 + (database[,54]==4)*as_lev4 + (database[,54]==5)*as_lev5 + (database[,89]==1.6)*as_mile16 + (database[,89]==4.4)*as_mile44 + (database[,89]==0.001)*as_mileunl;
  V[['pca']] = asc_pca + database[,60]*av_cost + database[,76]*av_app_fee + database[,86]*av_mo_fee + database[,66]*av_dr_rate + database[,6]*pca_hhs + database[,13]*pca_disab + database[,18]*pca_veh_own + (database[,13]<40)*pca_age + (database[,15]==1)*pca_male + (database[,20]==1)*pca_ft + (database[,20]==2)*pca_pt + database[,17]*pca_lic + database[,31]*pca_bike + database[,30]*pca_car + database[,32]*pca_smart + (database[,55]==3)*pa_lev3 + (database[,56]==3)*as_lev3 + (database[,55]==4)*pa_lev4 + (database[,56]==4)*as_lev4 + (database[,55]==5)*pa_lev5 + (database[,56]==5)*as_lev5 + (database[,91]==1.6)*pca_mile16 + (database[,91]==4.4)*pca_mile44 + (database[,91]==0.001)*pca_mileunl;
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(pc=1, pa=2, as=3, pca=4),
    avail        = list(pc=1, pa=1, as=1, pca=1),
    choiceVar     = database[,50],
    V             = V
  )
  ### Compute probabilities using MNL model
  P[['model']] = apollo_mnl(mnl_settings, functionality)
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}
# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

start_est <- Sys.time()
model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #
# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #
apollo_modelOutput(model)

end_est <- Sys.time()

tot_est <- end_est - start_est

ESS = ess(model$A)