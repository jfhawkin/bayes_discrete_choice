# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

start_est <- Sys.time()

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName ="SwissMetroMX",
  modelDescr ="Mixed logit model on Swiss mode choice data, uncorrelated normals in utility space - MSL",
  indivID   ="ID",  
  mixing    = TRUE, 
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
apollo_beta = c(mu_log_b_tt    = -1,
                sigma_log_b_tt = 0,
                b_tc    = -1,
                asc_train    = 0,
                asc_car = 0)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
# apollo_fixed = c("asc_sm")
apollo_fixed = c()

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  interDrawsType = "halton",
  interNDraws    = 1000,
  interUnifDraws = c(),
  interNormDraws = c("draws_tt"),
  intraDrawsType = "halton",
  intraNDraws    = 0,
  intraUnifDraws = c(),
  intraNormDraws = c()
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()
  
  randcoeff[["b_tt"]] = -exp(mu_log_b_tt + sigma_log_b_tt * draws_tt)
  
  return(randcoeff)
}

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
  V[['sm']]  = b_tt  * SM_TT_SCALED  + b_tc * SM_COST_SCALED
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
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Average across inter-individual draws
  P = apollo_avgInterDraws(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}
# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #
model = apollo_estimate(apollo_beta, apollo_fixed,
                        apollo_probabilities, apollo_inputs, 
                        estimate_settings=list(hessianRoutine="maxLik"))

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #
# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #
apollo_modelOutput(model)

end_est <- Sys.time()

tot_est <- end_est - start_est