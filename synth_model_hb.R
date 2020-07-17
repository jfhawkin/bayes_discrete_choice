# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
# rm(list = ls())

start_tot <- Sys.time()

### Load Apollo library
library(apollo)
library(mcmcse)
library(dplyr)
library(rstan)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="SynthHB",
  modelDescr ="Mixed logit model on synth database - HB",
  indivID   = "indID",  
  HB         = TRUE,
  nCores    = 4
)

# ################################################################# #
#### LOAD databaseA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #
database = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/synth_data_wide.csv',header = TRUE)

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(br_1=0,br_2=0,br_3=0,br_4=0,
                bf_1=0,bf_2=0,bf_3=0,bf_4=0,bf_5=0,bf_6=0,bf_7=0)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### HB settings                                                 ####
# ################################################################# #

apollo_HB = list(
  hbDist      = c(br_1="N",br_2="N",br_3="N",br_4="N",
                  bf_1="F",bf_2="F",bf_3="F",bf_4="F",bf_5="F",bf_6="F",bf_7="F"
				  ),
  gNCREP      = 25000, 
  gNEREP      = 25000, 
  gINFOSKIP   = 250,
  gFULLCV     = TRUE,
  hIW         = TRUE,
  priorVariance = 5,
  nodiagnostics = TRUE
)

# ################################################################# #
#### GROUP AND VALIdatabaseE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### ANALYSIS OF CHOICES                                         ####
# ################################################################# #
choiceAnalysis_settings <- list(
    alternatives  = c(a=0, b=1, c=2, d=3, e=4),
    avail        = list(a=1, b=1, c=1, d=1, e=1),
    choiceVar    = database$chosen
    # explanators  = database$altID
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
  V[['a']] = bf_1*database$xFix1_0+bf_2*database$xFix2_0+bf_3*database$xFix3_0+bf_4*database$xFix4_0+bf_5*database$xFix5_0+bf_6*database$xFix6_0+bf_7*database$xFix7_0 +
             br_1*database$xRnd1_0+br_2*database$xRnd2_0+br_3*database$xRnd3_0+br_4*database$xRnd4_0;
  V[['b']] = bf_1*database$xFix1_1+bf_2*database$xFix2_1+bf_3*database$xFix3_1+bf_4*database$xFix4_1+bf_5*database$xFix5_1+bf_6*database$xFix6_1+bf_7*database$xFix7_1 +
    br_1*database$xRnd1_1+br_2*database$xRnd2_1+br_3*database$xRnd3_1+br_4*database$xRnd4_1;
  V[['c']] = bf_1*database$xFix1_2+bf_2*database$xFix2_2+bf_3*database$xFix3_2+bf_4*database$xFix4_2+bf_5*database$xFix5_2+bf_6*database$xFix6_2+bf_7*database$xFix7_2 +
    br_1*database$xRnd1_2+br_2*database$xRnd2_2+br_3*database$xRnd3_2+br_4*database$xRnd4_2;
  V[['d']] = bf_1*database$xFix1_3+bf_2*database$xFix2_3+bf_3*database$xFix3_3+bf_4*database$xFix4_3+bf_5*database$xFix5_3+bf_6*database$xFix6_3+bf_7*database$xFix7_3 +
    br_1*database$xRnd1_3+br_2*database$xRnd2_3+br_3*database$xRnd3_3+br_4*database$xRnd4_3;
  V[['e']] = bf_1*database$xFix1_4+bf_2*database$xFix2_4+bf_3*database$xFix3_4+bf_4*database$xFix4_4+bf_5*database$xFix5_4+bf_6*database$xFix6_4+bf_7*database$xFix7_4 +
    br_1*database$xRnd1_4+br_2*database$xRnd2_4+br_3*database$xRnd3_4+br_4*database$xRnd4_4;
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(a=0, b=1, c=2, d=3, e=4),
    avail        = list(a=1, b=1, c=1, d=1, e=1),
    choiceVar     = database$chosen,
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

samples = cbind(model$A,model$F)
dim(samples) = c(nrow(samples),1,ncol(samples))
stan_results = monitor(samples,
                       warmup = 0,
                       probs = c(0.025, 0.25, 0.5, 0.75, 0.975),
                       digits_summary = 5)