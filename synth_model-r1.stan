data {
  int T; // number of choice tasks
  int I; // number of Individuals / number of rows
  int C; // number of columns in data matrix
  int K; // number of alternatives
  int PR; // number of random covariates with normal priors
  int PF; // number of fixed covariates with normal priors
  int P; // number of total random covariates
  int IT; // number of choice scenarios given as individuals x tasks per individual
  matrix[IT,K] choice; // binary indicator for choice
  matrix[IT, C] X; // data matrix
  int<lower=1> ind_label[IT] ;
}

// Following: https://mc-stan.org/docs/2_23/stan-users-guide/multivariate-hierarchical-priors-section.html
// for the multivariate component
parameters {
  vector[PF] betan; // hypermeans of the part-worths for fixed parameters with normal priors
  matrix[PR, I] z; // individual random effects (unscaled) (standardized component) for random parameters
  vector<lower=0>[PR] tau; // prior scale
  cholesky_factor_corr[PR] L_Omega; // prior correlation
  vector[PR] gamma; // random coeffs
}
transformed parameters {
  matrix[I, PR] beta_ind;
  beta_ind = rep_matrix(gamma', I) + (diag_pre_multiply(tau,L_Omega) * z)';
}

model {
  // create a temporary holding vector
  vector[I*T] log_prob; // we will add the log_prob for each scenario and pass it as a single vector to the posterior estimation engine
  vector[K] utils; // vector of utilities for each alternative
  row_vector[K] ones = rep_row_vector(1, K);
  
  // priors on the parameters
  tau ~ normal(0, 2);
  to_vector(z) ~ std_normal();
  betan ~ normal(0, 2);
  L_Omega ~ lkj_corr_cholesky(2);
  to_vector(gamma) ~ normal(0, 2);

  // log probabilities of each choice in the dataset
  for(i in 1:IT) {
      utils[1] = betan[1]*X[i,5]+betan[2]*X[i,6]+betan[3]*X[i,7]+betan[4]*X[i,8]+betan[5]*X[i,9]+betan[6]*X[i,10]+betan[7]*X[i,11]+
           beta_ind[ind_label[i],1]*X[i,12]+beta_ind[ind_label[i],2]*X[i,13]+beta_ind[ind_label[i],3]*X[i,14]+beta_ind[ind_label[i],4]*X[i,15];
      utils[2] = betan[1]*X[i,16]+betan[2]*X[i,17]+betan[3]*X[i,18]+betan[4]*X[i,19]+betan[5]*X[i,20]+betan[6]*X[i,21]+betan[7]*X[i,22]+
           beta_ind[ind_label[i],1]*X[i,23]+beta_ind[ind_label[i],2]*X[i,24]+beta_ind[ind_label[i],3]*X[i,25]+beta_ind[ind_label[i],4]*X[i,26];
      utils[3] = betan[1]*X[i,27]+betan[2]*X[i,28]+betan[3]*X[i,29]+betan[4]*X[i,30]+betan[5]*X[i,31]+betan[6]*X[i,32]+betan[7]*X[i,33]+
           beta_ind[ind_label[i],1]*X[i,34]+beta_ind[ind_label[i],2]*X[i,35]+beta_ind[ind_label[i],3]*X[i,36]+beta_ind[ind_label[i],4]*X[i,37];
      utils[4] = betan[1]*X[i,38]+betan[2]*X[i,39]+betan[3]*X[i,40]+betan[4]*X[i,41]+betan[5]*X[i,42]+betan[6]*X[i,43]+betan[7]*X[i,44]+
           beta_ind[ind_label[i],1]*X[i,45]+beta_ind[ind_label[i],2]*X[i,46]+beta_ind[ind_label[i],3]*X[i,47]+beta_ind[ind_label[i],4]*X[i,48];
      utils[5] = betan[1]*X[i,49]+betan[2]*X[i,50]+betan[3]*X[i,51]+betan[4]*X[i,52]+betan[5]*X[i,53]+betan[6]*X[i,54]+betan[7]*X[i,55]+
           beta_ind[ind_label[i],1]*X[i,56]+beta_ind[ind_label[i],2]*X[i,57]+beta_ind[ind_label[i],3]*X[i,58]+beta_ind[ind_label[i],4]*X[i,59];
      log_prob[i] = ones * (log_softmax(utils) .* choice[i,1:K]');
  }
  
  target += log_prob';
}
