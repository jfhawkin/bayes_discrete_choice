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
initf <- function() {
  # Fit the model with vb
  model_vi = vb(compiled_model, data = data_list, algorithm='meanfield')
  list(betan = as.list(get_posterior_mean(model_vi, pars=c('betan'))), z = as.list(get_posterior_mean(model_vi, pars=c('z'))), tau_unif=as.list(get_posterior_mean(model_vi, pars=c('tau_unif'))), L_Omega=as.list(get_posterior_mean(model_vi, pars=c('L_Omega'))), gamma=as.list(get_posterior_mean(model_vi, pars=c('gamma'))))
}

# Fit the model
model_fit = sampling(compiled_model, data = data_list, init=initf, iter = 2000, chains=1)

end_est <- Sys.time()

tot_est <- end_est - start_est

# sampler_params = get_sampler_params(model_fit, inc_warmup = TRUE)
# summary(do.call(rbind, sampler_params), digits = 2)
print(model_fit)

monitor(extract(model_fit, pars= c('betan','gamma','L_Omega','tau'), include=TRUE),digits_summary = 5)