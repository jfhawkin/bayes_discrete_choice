library(rstan) # observe startup messages
library(dplyr)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

dat = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/synth_data_wide.csv',header = TRUE)

# We have 8 tasks per individual for 184 individuals comparing 4 alternatives
I = 500  # Number of individuals
C = 59  # Number of data columns
K = 5    # Number of alternatives in each choice task
T = 5    # Number of alternatives by number of choice tasks
PR = 4 # One for each random parameter
PF = 7 # Number of alternatives/ASC 
P = PR + PF # Total number of covariates
IT = I*T # Number of individuals x number of tasks

X = data.matrix(dat)

choice = matrix(0L, nrow = IT, ncol = K)
for (k in 1:K){
  choice[,k] = as.integer(X[,4]==(k-1))
}

data_list = list(I = I,
                 C = C,
                 K = K,
                 T = T,
                 PR = PR,
                 PF = PF,
                 P = P,
                 IT = IT,
                 X = X,
                 ind_label = as.numeric(factor(dat$indID)),
                 choice = choice
)

# Compile the model
compiled_model = stan_model("synth_model.stan")

start_est <- Sys.time()

# Fit the model
model_opt = optimizing(compiled_model, init=0, , data = data_list)

print(model_opt)

end_est <- Sys.time()

tot_est <- end_est - start_est