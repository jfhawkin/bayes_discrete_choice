data {
  int N; // number of rows
  int T; // number of inidvidual-choice sets/task combinations
  int I; // number of Individuals
  int P; // number of non-random covariates
  int P2; // number of random covariates
  
  vector<lower = 0, upper = 1>[N] choice; // binary indicator for choice
  matrix[N, P] X; // non-random attributes
  matrix[N, P2] X2; // random attributes
  vector[N] av; // indicator of availability for each alternative
  
  int task[T]; // index for tasks
  int task_individual[T]; // index for individual
  int start[T]; // the starting observation for each task
  int end[T]; // the ending observation for each task
}
parameters {
  vector[P] beta; // hypermeans of the part-worths for non-random parameters
  vector[P2] beta2; // hypermeans of the part-worths for random parameters
  vector<lower=0,upper=pi()/2>[P2] tau_unif; // prior scale
  matrix[P2, I] z; // individual random effects (unscaled) (standardized component)
}
transformed parameters {
  // here we use the reparameterization discussed on slide 30
  vector<lower=0>[P2] tau;     // prior scale
  matrix[I, P2] beta_ind;
  matrix[I, P] beta_fixed;
  for (k in 1:P2) tau[k] = 2.5 * tan(tau_unif[k]);
  beta_ind = rep_matrix(beta2', I) + (diag_matrix(tau) * z)';
  beta_fixed = rep_matrix(beta', I);
}
model {
  // create a temporary holding vector
  vector[N] log_prob;
  
  // priors on the parameters
  beta ~ normal(0, 10);
  beta2 ~ normal(-1, 10);
  to_vector(z) ~ normal(0, 1);
  
  // log probabilities of each choice in the dataset
  for(t in 1:T) {
    log_prob[start[t]:end[t]] = av[start[t]:end[t]] .* (log(softmax(X[start[t]:end[t]]*beta_fixed[task_individual[t]]' + X2[start[t]:end[t]]*beta_ind[task_individual[t]]')));
  }
  
  // use the likelihood derivation on slide 29
  target += log_prob' * choice;
}
