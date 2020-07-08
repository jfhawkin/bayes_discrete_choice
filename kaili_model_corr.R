library(rstan) # observe startup messages
library(dplyr)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

dat = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/xmatswave2019.csv',header = FALSE)

# We have 8 tasks per individual for 184 individuals comparing 4 alternatives
I = 184  # Number of individuals
C = 424  # Number of data columns
K = 4    # Number of alternatives in each choice task
T = 8    # Number of alternatives by number of choice tasks
PN = 52 # One for each parameter
ASC = 3 # Number of ASC 
P = ASC + PN # Total number of covariates

X = data.matrix(dat)

# Scale one time costs and monthly mileage to $1000/1000 mi
X[,71] = X[,71] / 1000
X[,73] = X[,73] / 1000
X[,102] = X[,102] / 1000
X[,104] = X[,104] / 1000
X[,116] = X[,116] / 1000
X[,118] = X[,118] / 1000
X[,147] = X[,147] / 1000
X[,149] = X[,149] / 1000
X[,160] = X[,160] / 1000
X[,161] = X[,161] / 1000
X[,163] = X[,163] / 1000
X[,192] = X[,194] / 1000
X[,205] = X[,205] / 1000
X[,206] = X[,206] / 1000
X[,208] = X[,208] / 1000
X[,237] = X[,237] / 1000
X[,251] = X[,251] / 1000
X[,253] = X[,253] / 1000
X[,284] = X[,284] / 1000
X[,295] = X[,295] / 1000
X[,296] = X[,296] / 1000
X[,298] = X[,298] / 1000
X[,312] = X[,312] / 1000
X[,327] = X[,327] / 1000
X[,340] = X[,340] / 1000
X[,341] = X[,341] / 1000
X[,343] = X[,343] / 1000
X[,372] = X[,371] / 1000
X[,374] = X[,374] / 1000
X[,385] = X[,385] / 1000
X[,386] = X[,386] / 1000
X[,388] = X[,388] / 1000
X[,417] = X[,417] / 1000
X[,419] = X[,419] / 1000
X[,87] =  X[,87] / 1000
X[,134] = X[,134] / 1000
X[,160] = X[,160] / 1000
X[,179] = X[,179] / 1000
X[,222] = X[,222] / 1000
X[,267] = X[,267] / 1000
X[,357] = X[,357] / 1000
X[,402] = X[,402] / 1000
# Scale monthly costs to $100
X[,90] = X[,90] / 100
X[,91] = X[,91] / 100
X[,93] = X[,93] / 100
X[,97] = X[,97] / 100
X[,135] = X[,135] / 100
X[,136] = X[,136] / 100
X[,138] = X[,138] / 100
X[,142] = X[,142] / 100
X[,180] = X[,180] / 100
X[,181] = X[,181] / 100
X[,183] = X[,183] / 100
X[,187] = X[,187] / 100
X[,224] = X[,224] / 100
X[,225] = X[,225] / 100
X[,226] = X[,226] / 100
X[,228] = X[,228] / 100
X[,232] = X[,232] / 100
X[,269] = X[,269] / 100
X[,270] = X[,270] / 100
X[,271] = X[,271] / 100
X[,273] = X[,273] / 100
X[,277] = X[,277] / 100
X[,315] = X[,315] / 100
X[,316] = X[,316] / 100
X[,318] = X[,318] / 100
X[,322] = X[,322] / 100
X[,359] = X[,359] / 100
X[,360] = X[,360] / 100
X[,361] = X[,361] / 100
X[,363] = X[,363] / 100
X[,367] = X[,367] / 100
X[,405] = X[,405] / 100
X[,406] = X[,406] / 100
X[,408] = X[,408] / 100
X[,412] = X[,412] / 100

choice = matrix(0L, nrow = I, ncol = T*K)
for(i in 1:T){
  for (j in 1:K){
    choice[,j+(i-1)*K] = as.integer(X[,49+(i-1)]==j)
  }
}

data_list = list(I = I,
                 C = C,
                 K = K,
                 T = T,
                 PN = PN,
                 ASC = ASC,
                 P = P,
                 X = X,
                 choice = choice
)

# Compile the model
compiled_model = stan_model("kaili_model_corr.stan")

# Fit the model
model_fit = sampling(compiled_model, data = data_list, iter = 2000, chains=4)


sampler_params = get_sampler_params(model_fit, inc_warmup = TRUE)
summary(do.call(rbind, sampler_params), digits = 2)
print(model_fit)