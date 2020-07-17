# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

start_est <- Sys.time()

### Load Apollo library
library(apollo)
library(mcmcse)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="SwissMetroHB",
  modelDescr ="Mixed logit model on Swiss mode choice data, uncorrelated Lognormals in utility space - HB",
  indivID   ="ID",  
  HB         = TRUE,
  nCores    = 4
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

database = read.csv("https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/swissmetro.dat",sep="\t",header=TRUE)
database = subset(database,database$PURPOSE==1 | database$PURPOSE==3 | database$CHOICE!=0)

# Scale the time and costs
database$TRAIN_TT_SCALED = database$TRAIN_TT / 100
database$SM_TT_SCALED = database$SM_TT / 100
database$CAR_TT_SCALED = database$CAR_TT / 100
database$TRAIN_COST = database$TRAIN_CO * (database$GA == 0)
database$SM_COST = database$SM_CO * (database$GA == 0)
database$TRAIN_COST_SCALED = database$TRAIN_COST / 100
database$SM_COST_SCALED = database$SM_COST / 100
database$CAR_COST_SCALED = database$CAR_CO / 100

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(b_tt    =0,
                b_tc    =0,
                asc_train    = 0,
                asc_car = 0,
                asc_sm = 0)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_sm")

# ################################################################# #
#### HB settings                                                 ####
# ################################################################# #

apollo_HB = list(
  hbDist      = c(b_tt="LN-", b_tc="F", asc_train="F",
                  asc_car="F", asc_sm="F"),
  gNCREP      = 1000, 
  gNEREP      = 1000, 
  gINFOSKIP   = 250,
  gFULLCV     = FALSE,
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
  alternatives = c(train=1, sm=2, car=3),
  avail        = list(train=database$TRAIN_AV, sm=database$SM_AV, car=database$CAR_AV),
  choiceVar    = database$CHOICE,
  explanators  = database[,c("MALE","AGE","INCOME")]
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
  V[['train']]  = asc_train  + b_tt * TRAIN_TT_SCALED + b_tc * TRAIN_COST_SCALED
  V[['sm']]  = asc_sm + b_tt  * SM_TT_SCALED  + b_tc * SM_COST_SCALED
  V[['car']]  = asc_car  + b_tt  * CAR_TT_SCALED  + b_tc * CAR_COST_SCALED
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(train=1, sm=2, car=3),
    avail         = list(train=TRAIN_AV, sm=SM_AV, car=CAR_AV),
    choiceVar     = CHOICE,
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

samples = cbind(model$A,model$F)
dim(samples) = c(nrow(samples),1,ncol(samples))
stan_results = monitor(samples,
                       warmup = 0,
                       probs = c(0.025, 0.25, 0.5, 0.75, 0.975),
                       digits_summary = 5)