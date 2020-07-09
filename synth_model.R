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
PF = 7 # Number of fixed parameters
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
                 choice = choice
)

# Compile the model
compiled_model = stan_model("synth_model.stan")

# Fit the model
model_fit = sampling(compiled_model, data = data_list, init=0, iter = 50000, chains=2)

# sampler_params = get_sampler_params(model_fit, inc_warmup = TRUE)
# summary(do.call(rbind, sampler_params), digits = 2)
print(model_fit)

summary(model_fit, pars = c("gamma"))$summary
summary(model_fit, pars = c("lp__"))$summary
