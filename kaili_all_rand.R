library(rstan) # observe startup messages
library(dplyr)
library(dummies)
rstan_options(auto_write = TRUE)

dat = read.csv('https://raw.githubusercontent.com/jfhawkin/bayes_discrete_choice/master/xmatswave2019.csv',header = FALSE)
# dat = cbind(dat, dummy(dat$ALT, sep = "_"))

# We have 8 tasks per individual for 184 individuals comparing 4 alternatives
I = 184  # Number of individuals
C = 424  # Number of data columns
K = 4    # Number of alternatives in each choice task
T = 8    # Number of alternatives by number of choice tasks
PN = 1   # One each for ASC, automation level
PLN = 1 # One each for monthly mileage
PNLN = 5 # remaining variables
ASC = 4 # Number of alternatives/ASC 
P = ASC + PN + PLN + PNLN # Total number of covariates

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
                 PLN = PLN,
                 PNLN = PNLN,
                 ASC = ASC,
                 P = P,
                 X = X,
                 choice = choice
)

# Compile the model
compiled_model = stan_model("kaili_all_rand.stan")

# Fit the model
model_fit = sampling(compiled_model, data = data_list, iter = 3000, cores=4, chains=4, control = list(adapt_delta = 0.95))

sampler_params = get_sampler_params(model_fit, inc_warmup = TRUE)
summary(do.call(rbind, sampler_params), digits = 2)
print(model_fit)

# accept_stat__    stepsize__       treedepth__    n_leapfrog__   divergent__      energy__     
# Min.   :0.00   Min.   :5.4e-04   Min.   : 0.0   Min.   :   1   Min.   :0.00   Min.   :  3639  
# 1st Qu.:0.95   1st Qu.:7.7e-03   1st Qu.: 7.0   1st Qu.: 220   1st Qu.:0.00   1st Qu.:  3797  
# Median :0.98   Median :9.4e-03   Median : 8.0   Median : 336   Median :0.00   Median :  3832  
# Mean   :0.94   Mean   :1.4e-02   Mean   : 7.9   Mean   : 403   Mean   :0.44   Mean   :  4484  
# 3rd Qu.:0.99   3rd Qu.:1.3e-02   3rd Qu.: 9.0   3rd Qu.: 511   3rd Qu.:1.00   3rd Qu.:  3865  
# Max.   :1.00   Max.   :1.1e+01   Max.   :10.0   Max.   :1023   Max.   :1.00   Max.   :965618  
# > print(model_fit)
# Inference for Stan model: kaili_all_rand.
# 4 chains, each with iter=3000; warmup=1500; thin=1; 
# post-warmup draws per chain=1500, total post-warmup draws=6000.
# 
# mean se_mean    sd     2.5%      25%      50%      75%    97.5% n_eff Rhat
# betan[1]              -0.37    0.00  0.04    -0.45    -0.39    -0.37    -0.34    -0.29  2573 1.00
# betaln[1]              0.00    0.00  0.00     0.00     0.00     0.00     0.00     0.00  6028 1.00
# betanln[1]             0.17    0.00  0.02     0.13     0.16     0.17     0.19     0.22   908 1.01
# betanln[2]             0.02    0.00  0.01     0.00     0.01     0.02     0.03     0.05  3021 1.00
# betanln[3]             0.01    0.00  0.01     0.00     0.01     0.01     0.02     0.03  3426 1.00
# betanln[4]             0.02    0.00  0.00     0.02     0.02     0.02     0.03     0.03  3078 1.00
# betanln[5]             0.31    0.00  0.05     0.20     0.27     0.30     0.34     0.41  1976 1.00
# z[1,1]                 0.13    0.02  0.98    -1.79    -0.53     0.13     0.78     2.04  2951 1.00
# z[1,2]                -0.05    0.02  1.01    -2.02    -0.77    -0.04     0.63     1.97  2866 1.00
# z[1,3]                 0.31    0.01  0.75    -1.01    -0.23     0.26     0.82     1.87  2847 1.00
# z[1,4]                 0.07    0.02  1.00    -1.90    -0.63     0.07     0.77     1.99  3454 1.00
# z[1,5]                -0.08    0.02  0.99    -2.02    -0.77    -0.08     0.57     1.86  3326 1.00
# z[1,6]                -0.11    0.02  1.01    -2.11    -0.81    -0.12     0.58     1.86  2784 1.00
# z[1,7]                -0.63    0.01  0.74    -2.06    -1.13    -0.62    -0.13     0.82  2792 1.00
# z[2,1]                 0.09    0.02  0.98    -1.79    -0.57     0.06     0.75     2.05  3903 1.00
# z[2,2]                 0.00    0.02  1.01    -1.95    -0.72    -0.02     0.68     2.03  2543 1.00
# z[2,3]                -0.03    0.02  0.75    -1.29    -0.54    -0.10     0.43     1.65  2167 1.00
# z[2,4]                -0.03    0.02  0.98    -2.02    -0.67    -0.02     0.61     1.89  3488 1.00
# z[2,5]                 0.05    0.02  0.99    -1.91    -0.60     0.07     0.72     1.94  3464 1.00
# z[2,6]                -0.12    0.02  0.98    -2.09    -0.80    -0.12     0.51     1.85  3420 1.00
# z[2,7]                -0.79    0.01  0.85    -2.47    -1.34    -0.80    -0.22     0.89  3363 1.00
# z[3,1]                -0.10    0.02  0.98    -2.02    -0.76    -0.09     0.57     1.78  3223 1.00
# z[3,2]                -0.04    0.02  1.02    -2.02    -0.73    -0.03     0.65     1.93  3227 1.00
# z[3,3]                 0.56    0.01  0.81    -0.91     0.00     0.51     1.08     2.23  2974 1.00
# z[3,4]                -0.01    0.02  1.01    -1.97    -0.70    -0.02     0.68     1.98  3138 1.00
# z[3,5]                 0.04    0.02  1.02    -2.00    -0.67     0.06     0.73     2.02  3325 1.00
# z[3,6]                 0.01    0.02  0.99    -1.91    -0.67     0.02     0.69     1.89  3047 1.00
# z[3,7]                 0.40    0.01  0.83    -1.25    -0.14     0.39     0.97     2.07  3318 1.00
# z[4,1]                 0.07    0.02  1.00    -1.86    -0.62     0.05     0.76     1.99  3324 1.00
# z[4,2]                -0.04    0.02  0.98    -1.99    -0.67    -0.04     0.61     1.92  3117 1.00
# z[4,3]                 0.44    0.01  0.82    -1.04    -0.13     0.40     0.98     2.14  2987 1.00
# z[4,4]                -0.10    0.02  0.98    -2.00    -0.78    -0.09     0.56     1.83  3393 1.00
# z[4,5]                -0.03    0.02  0.99    -1.96    -0.70    -0.04     0.63     1.96  3049 1.00
# z[4,6]                 0.02    0.02  1.00    -1.94    -0.65     0.04     0.71     1.94  2894 1.00
# z[4,7]                -0.64    0.01  0.83    -2.25    -1.20    -0.64    -0.08     0.99  3075 1.00
# z[5,1]                -0.06    0.02  1.01    -2.03    -0.73    -0.06     0.64     1.94  2990 1.00
# z[5,2]                 0.26    0.02  0.99    -1.67    -0.42     0.26     0.92     2.18  3024 1.00
# z[5,3]                -1.12    0.01  0.53    -2.12    -1.47    -1.14    -0.80     0.01  2975 1.00
# z[5,4]                -0.08    0.02  0.99    -1.99    -0.75    -0.10     0.60     1.87  2268 1.00
# z[5,5]                -0.05    0.02  1.01    -2.01    -0.75    -0.05     0.64     1.91  2901 1.00
# z[5,6]                 0.02    0.02  0.99    -1.87    -0.63     0.02     0.69     1.99  3238 1.00
# z[5,7]                -1.04    0.02  0.86    -2.79    -1.61    -1.04    -0.46     0.64  3309 1.00
# z[6,1]                -0.06    0.02  0.96    -1.97    -0.72    -0.06     0.62     1.80  3467 1.00
# z[6,2]                -0.06    0.02  0.97    -1.95    -0.73    -0.06     0.60     1.85  2899 1.00
# z[6,3]                 0.52    0.01  0.81    -0.96    -0.05     0.48     1.06     2.24  3150 1.00
# z[6,4]                -0.01    0.02  0.99    -1.98    -0.66    -0.04     0.64     1.93  3346 1.00
# z[6,5]                 0.02    0.02  1.04    -2.00    -0.69     0.04     0.71     2.00  3294 1.00
# z[6,6]                 0.03    0.02  1.01    -1.94    -0.67     0.04     0.74     1.97  2755 1.00
# z[6,7]                 0.44    0.01  0.83    -1.19    -0.11     0.42     0.99     2.09  3334 1.00
# z[7,1]                -0.07    0.02  0.97    -1.96    -0.73    -0.06     0.59     1.85  3276 1.00
# z[7,2]                 0.26    0.02  0.99    -1.67    -0.40     0.25     0.91     2.27  3353 1.00
# z[7,3]                -1.11    0.01  0.53    -2.14    -1.46    -1.14    -0.78     0.00  2742 1.00
# z[7,4]                -0.07    0.02  1.00    -2.02    -0.74    -0.08     0.60     1.91  3059 1.00
# z[7,5]                -0.01    0.02  0.98    -1.96    -0.68    -0.02     0.64     1.92  3387 1.00
# z[7,6]                -0.02    0.02  1.02    -2.10    -0.68    -0.02     0.65     1.98  2654 1.00
# z[7,7]                -1.04    0.01  0.83    -2.61    -1.63    -1.02    -0.49     0.60  3400 1.00
# z[8,1]                 0.15    0.02  1.03    -1.91    -0.52     0.16     0.84     2.13  3351 1.00
# z[8,2]                -0.06    0.02  0.99    -1.95    -0.72    -0.05     0.60     1.85  3783 1.00
# z[8,3]                 0.24    0.01  0.77    -1.12    -0.33     0.20     0.75     1.87  3210 1.00
# z[8,4]                 0.03    0.02  0.99    -1.89    -0.66     0.03     0.72     1.93  3840 1.00
# z[8,5]                -0.06    0.02  1.00    -2.04    -0.73    -0.05     0.61     1.85  3654 1.00
# z[8,6]                -0.13    0.02  1.03    -2.15    -0.81    -0.14     0.58     1.84  3298 1.00
# z[8,7]                -1.01    0.02  0.81    -2.55    -1.57    -1.02    -0.46     0.56  2860 1.00
# z[9,1]                 0.13    0.02  0.97    -1.77    -0.55     0.11     0.80     2.00  3117 1.00
# z[9,2]                -0.03    0.02  1.01    -2.08    -0.70    -0.02     0.65     1.94  2795 1.00
# z[9,3]                 0.39    0.01  0.82    -1.13    -0.19     0.35     0.93     2.09  3429 1.00
# z[9,4]                -0.05    0.02  1.00    -2.02    -0.73    -0.05     0.61     1.90  3367 1.00
# z[9,5]                -0.04    0.02  0.98    -1.92    -0.69    -0.06     0.60     1.91  3066 1.00
# z[9,6]                -0.06    0.02  0.99    -2.06    -0.69    -0.04     0.59     1.86  3036 1.00
# z[9,7]                -1.50    0.01  0.81    -3.08    -2.03    -1.48    -0.97     0.08  3350 1.00
# z[10,1]                0.11    0.02  0.98    -1.81    -0.54     0.11     0.75     2.03  3549 1.00
# z[10,2]               -0.04    0.02  0.99    -1.97    -0.70    -0.04     0.63     1.91  3019 1.00
# z[10,3]               -0.80    0.01  0.56    -1.83    -1.17    -0.85    -0.46     0.37  2933 1.00
# z[10,4]               -0.01    0.02  0.98    -1.95    -0.68    -0.02     0.66     1.89  2958 1.00
# z[10,5]                0.01    0.02  0.99    -1.92    -0.64     0.01     0.65     1.99  3652 1.00
# z[10,6]               -0.08    0.02  0.97    -2.01    -0.70    -0.08     0.57     1.84  2941 1.00
# z[10,7]                0.46    0.01  0.80    -1.09    -0.09     0.46     1.02     1.98  3022 1.00
# z[11,1]               -0.05    0.02  1.03    -2.02    -0.73    -0.05     0.62     2.00  2925 1.00
# z[11,2]               -0.02    0.02  0.98    -1.89    -0.70    -0.01     0.67     1.83  3319 1.00
# z[11,3]                0.09    0.01  0.71    -1.13    -0.43     0.03     0.55     1.62  2506 1.00
# z[11,4]                0.01    0.02  0.99    -1.96    -0.65     0.02     0.68     1.91  3058 1.00
# z[11,5]                0.04    0.02  0.99    -1.85    -0.65     0.04     0.70     1.97  3507 1.00
# z[11,6]               -0.01    0.02  0.97    -1.95    -0.62    -0.02     0.64     1.85  3240 1.00
# z[11,7]                0.46    0.02  0.80    -1.10    -0.10     0.46     1.01     2.03  2784 1.00
# z[12,1]               -0.07    0.02  0.99    -2.00    -0.71    -0.09     0.59     1.86  2839 1.00
# z[12,2]               -0.05    0.02  0.99    -2.02    -0.70    -0.04     0.62     1.83  3338 1.00
# z[12,3]                0.55    0.01  0.82    -0.94    -0.04     0.51     1.08     2.25  3394 1.00
# z[12,4]               -0.03    0.02  1.02    -2.04    -0.71    -0.02     0.63     1.98  3417 1.00
# z[12,5]                0.03    0.02  0.99    -1.93    -0.64     0.02     0.68     1.97  2963 1.00
# z[12,6]                0.01    0.02  0.99    -1.94    -0.66    -0.01     0.69     1.97  3282 1.00
# z[12,7]                0.40    0.01  0.83    -1.20    -0.15     0.40     0.98     1.99  3107 1.00
# z[13,1]               -0.04    0.02  1.01    -2.00    -0.72    -0.03     0.64     1.95  2852 1.00
# z[13,2]                0.25    0.02  1.00    -1.74    -0.43     0.25     0.92     2.19  3387 1.00
# z[13,3]               -1.12    0.01  0.53    -2.12    -1.47    -1.15    -0.80     0.04  2140 1.00
# z[13,4]               -0.04    0.02  1.03    -2.05    -0.74    -0.04     0.65     1.98  2882 1.00
# z[13,5]               -0.03    0.02  0.98    -1.95    -0.70    -0.04     0.64     1.91  2980 1.00
# z[13,6]               -0.02    0.02  1.00    -2.02    -0.68    -0.02     0.62     1.96  3366 1.00
# z[13,7]               -1.03    0.01  0.82    -2.60    -1.58    -1.03    -0.49     0.63  3149 1.00
# z[14,1]                0.10    0.02  1.01    -1.87    -0.57     0.08     0.77     2.11  3375 1.00
# z[14,2]                0.01    0.02  0.98    -1.90    -0.65     0.01     0.67     1.95  3636 1.00
# [ reached getOption("max.print") -- omitted 3998 rows ]
# 
# Samples were drawn using NUTS(diag_e) at Sat Jun  6 04:29:41 2020.
# For each parameter, n_eff is a crude measure of effective sample size,
# and Rhat is the potential scale reduction factor on split chains (at 
#                                                                   convergence, Rhat=1).