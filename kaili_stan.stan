data {
  int T; // number of choice tasks
  int I; // number of Individuals / number of rows
  int C; // number of columns in data matrix
  int K; // number of alternatives
  int P1; // number of non-random covariates for task1
  int P2; // number of non-random covariates for task1
  int P3; // number of non-random covariates for task1
  int P4; // number of non-random covariates for task1
  int PR; // number of random covariates
  
  matrix[I,(T*K)] choice; // binary indicator for choice
  matrix[I, C] X; // data matrix
}
parameters {
  vector[P1] beta1; // hypermeans of the part-worths for non-random parameters for alternative 1
  vector[P2] beta2; // hypermeans of the part-worths for non-random parameters for alternative 1
  vector[P3] beta3; // hypermeans of the part-worths for non-random parameters for alternative 1
  vector[P4] beta4; // hypermeans of the part-worths for non-random parameters for alternative 1
  vector[PR] betar; // hypermeans of the part-worths for random parameters
  vector<lower = 0>[PR] tau; // diagonal of the part-worth covariance matrix for random parameters (scale component)
  matrix[I, PR] z; // individual random effects (unscaled) (standardized component)
}
transformed parameters {
  matrix[I, PR] beta_individual = rep_matrix(betar', I) + z*diag_matrix(tau);
  matrix[I, P1] beta_fixed1 = rep_matrix(beta1', I);
  matrix[I, P2] beta_fixed2 = rep_matrix(beta2', I);
  matrix[I, P3] beta_fixed3 = rep_matrix(beta3', I);
  matrix[I, P4] beta_fixed4 = rep_matrix(beta4', I);
}
model {
  // create a temporary holding vector
  vector[I*T] log_prob; // we will add the log_prob for each scenario and pass it as a single vector to the posterior estimation engine
  vector[K] utils; // vector of utilities for each alternative
  int shift = 45; // integer shift between each iteration of j below (i.e., shift for number of attributes in each SP)
  int tshift; // total shift for each individual/task combination
  row_vector[K] ones = rep_row_vector(1, K);
  
  // priors on the parameters
  tau ~ inv_gamma(1, 1);
  beta1 ~ cauchy(0, 2.5);
  beta2 ~ cauchy(0, 2.5);
  beta3 ~ cauchy(0, 2.5);
  beta4 ~ cauchy(0, 2.5);
  betar ~ cauchy(0, 2.5);
  to_vector(z) ~ normal(0, 1);
  
  // log probabilities of each choice in the dataset
  // Model is: [additional cost ($1000) / driving rate ($)] + [driving cost ($/100km) / application fee] + [parking cost ($100/month) / subscription cost ($/month)] + [hierarchical AV level]
  for(i in 1:I) {
    for(j in 1:T){
        tshift = shift*(j-1);
        utils[1] = X[i,(tshift+70)]*beta_fixed1[i,1]+X[i,(tshift+80)]*beta_fixed1[i,2]+X[i,(tshift+90)]*beta_fixed1[i,3]+X[i,(tshift+65)]*beta_individual[i,1];
        utils[2] = X[i,(tshift+71)]*beta_fixed2[i,1]+X[i,(tshift+81)]*beta_fixed2[i,2]+X[i,(tshift+91)]*beta_fixed2[i,3]+X[i,(tshift+66)]*beta_individual[i,2];
        utils[3] = X[i,(tshift+77)]*beta_fixed3[i,1]+X[i,(tshift+87)]*beta_fixed3[i,2]+X[i,(tshift+97)]*beta_fixed3[i,3]+X[i,(tshift+67)]*beta_individual[i,3];
        utils[4] = X[i,(tshift+73)]*beta_fixed4[i,1]+X[i,(tshift+83)]*beta_fixed4[i,2]+X[i,(tshift+93)]*beta_fixed4[i,3]+X[i,(tshift+68)]*beta_individual[i,4];
        log_prob[i+(j-1)*I] = ones * (log_softmax(utils) .* choice[i,(1+(j-1)):(K+(j-1))]');
    }
  }
  
  target += log_prob';
}
