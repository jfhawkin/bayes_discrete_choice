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
transformed data{
	vector[C] outcome_mean;
	vector[C] outcome_sd;
	matrix[IT,C] scaled_outcome;
	
	// outcome_mean: mean outcome value
	for (i in 1:C) outcome_mean[i] = mean(X[,i]);
	// outcome_sd: sd of outcomes
	for (i in 1:C) outcome_sd[i] = sd(X[,i]);
	
	// scaled_outcome: outcomes scaled to have zero mean and unit variance
	scaled_outcome = (X-rep_matrix(outcome_mean',IT))./rep_matrix(outcome_sd',IT) ;

}

// Following: https://mc-stan.org/docs/2_23/stan-users-guide/multivariate-hierarchical-priors-section.html
// for the multivariate component
parameters {
  vector[PF] betan; // hypermeans of the part-worths for fixed parameters with normal priors
  matrix[PR, I] z; // individual random effects (unscaled) (standardized component) for random parameters
  cholesky_factor_corr[PR] L_Omega; // prior correlation
  vector[PR] gamma; // random coeffs
}
transformed parameters {
  matrix[I, PR] beta_ind;
  beta_ind = rep_matrix(gamma', I) + (L_Omega * z)';
}

model {
  // create a temporary holding vector
  vector[I*T] log_prob; // we will add the log_prob for each scenario and pass it as a single vector to the posterior estimation engine
  vector[K] utils; // vector of utilities for each alternative
  int ind=1; // individual id for a given IT to translate to individual-level heterogeneity for multi-task dataset
  row_vector[K] ones = rep_row_vector(1, K);
  
  // priors on the parameters
  to_vector(z) ~ std_normal();
  betan ~ std_normal();
  L_Omega ~ lkj_corr_cholesky(2);
  to_vector(gamma) ~ std_normal();

  // log probabilities of each choice in the dataset
  for(i in 1:IT) {
      utils[1] = betan[1]*scaled_outcome[i,5]+betan[2]*scaled_outcome[i,6]+betan[3]*scaled_outcome[i,7]+betan[4]*scaled_outcome[i,8]+betan[5]*scaled_outcome[i,9]+betan[6]*scaled_outcome[i,10]+betan[7]*scaled_outcome[i,11]+
           beta_ind[ind_label[i],1]*scaled_outcome[i,12]+beta_ind[ind_label[i],2]*scaled_outcome[i,13]+beta_ind[ind_label[i],3]*scaled_outcome[i,14]+beta_ind[ind_label[i],4]*scaled_outcome[i,15];
      utils[2] = betan[1]*scaled_outcome[i,16]+betan[2]*scaled_outcome[i,17]+betan[3]*scaled_outcome[i,18]+betan[4]*scaled_outcome[i,19]+betan[5]*scaled_outcome[i,20]+betan[6]*scaled_outcome[i,21]+betan[7]*scaled_outcome[i,22]+
           beta_ind[ind_label[i],1]*scaled_outcome[i,23]+beta_ind[ind_label[i],2]*scaled_outcome[i,24]+beta_ind[ind_label[i],3]*scaled_outcome[i,25]+beta_ind[ind_label[i],4]*scaled_outcome[i,26];
      utils[3] = betan[1]*scaled_outcome[i,27]+betan[2]*scaled_outcome[i,28]+betan[3]*scaled_outcome[i,29]+betan[4]*scaled_outcome[i,30]+betan[5]*scaled_outcome[i,31]+betan[6]*scaled_outcome[i,32]+betan[7]*scaled_outcome[i,33]+
           beta_ind[ind_label[i],1]*scaled_outcome[i,34]+beta_ind[ind_label[i],2]*scaled_outcome[i,35]+beta_ind[ind_label[i],3]*scaled_outcome[i,36]+beta_ind[ind_label[i],4]*scaled_outcome[i,37];
      utils[4] = betan[1]*scaled_outcome[i,38]+betan[2]*scaled_outcome[i,39]+betan[3]*scaled_outcome[i,40]+betan[4]*scaled_outcome[i,41]+betan[5]*scaled_outcome[i,42]+betan[6]*scaled_outcome[i,43]+betan[7]*scaled_outcome[i,44]+
           beta_ind[ind_label[i],1]*scaled_outcome[i,45]+beta_ind[ind_label[i],2]*scaled_outcome[i,46]+beta_ind[ind_label[i],3]*scaled_outcome[i,47]+beta_ind[ind_label[i],4]*scaled_outcome[i,48];
      utils[5] = betan[1]*scaled_outcome[i,49]+betan[2]*scaled_outcome[i,50]+betan[3]*scaled_outcome[i,51]+betan[4]*scaled_outcome[i,52]+betan[5]*scaled_outcome[i,53]+betan[6]*scaled_outcome[i,54]+betan[7]*scaled_outcome[i,55]+
           beta_ind[ind_label[i],1]*scaled_outcome[i,56]+beta_ind[ind_label[i],2]*scaled_outcome[i,57]+beta_ind[ind_label[i],3]*scaled_outcome[i,58]+beta_ind[ind_label[i],4]*scaled_outcome[i,59];
      log_prob[i] = ones * (log_softmax(utils) .* choice[i,1:K]');
  }
  
  target += log_prob';
}
