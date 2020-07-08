### Clear memory
rm(list = ls())

start_est <- Sys.time()
library(rstan) # observe startup messages
library(dplyr)
library(dummies)
rstan_options(auto_write = TRUE)

dat = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/swissmetro_long.csv')
dat = filter(dat, (PURPOSE==1 | PURPOSE == 3) & CHOICE_OLD!=0)
dat <- cbind(dat, dummy(dat$ALT, sep = "_"))

dat$TT = dat$TT / 100
dat$COST = dat$COST / 100

# We have nine tasks per individual for 752 individuals comparing 3 choices
I <- 752
T <- I*9
K <- 3
P <- 3
P2 <- 1 # We'll vary one mode attribute for each choice at each task
N <- T*K

X <- data.matrix(dat[c('dat_1', 'dat_3','COST')])
X2 <- data.matrix(dat$TT)
av <- dat$AV

indexes <- data_frame(individual = rep(1:I, each = K*9),
                      task = rep(1:T, each = K),
                      row = 1:(T*K)) %>%
  group_by(task) %>% 
  summarise(task_individual = first(individual),
            start = first(row),
            end = last(row)) 

choice <- dat$CHOICE

data_list <- list(N = N, 
                  I = I, 
                  P = P, 
                  P2 = P2, 
                  K = K, 
                  T = T, 
                  X = X, 
                  X2 = X2, 
                  choice = choice,
                  av = av,
                  start = indexes$start,
                  end = indexes$end, 
                  task_individual = indexes$task_individual,
                  task = indexes$task)

# Compile the model
compiled_model <- stan_model("swiss_metro.stan")

# Fit the model
model_fit <- sampling(compiled_model, data = data_list, iter = 2000, cores=4, chains=4)

sampler_params <- get_sampler_params(model_fit, inc_warmup = TRUE)
summary(do.call(rbind, sampler_params), digits = 2)
print(model_fit)

end_est <- Sys.time()

tot_est <- end_est - start_est