data {
  int T; // number of choice tasks
  int I; // number of Individuals / number of rows
  int C; // number of columns in data matrix
  int K; // number of alternatives
  int PN; // number of random covariates with normal priors
  int ASC; // number of alternatives/ASC
  int P; // number of total random covariates
  matrix[I,(T*K)] choice; // binary indicator for choice
  matrix[I, C] X; // data matrix
}

// Following: https://mc-stan.org/docs/2_23/stan-users-guide/multivariate-hierarchical-priors-section.html
// for the multivariate component
parameters {
  vector[PN] betan; // hypermeans of the part-worths for random parameters with normal priors (less ASC)
  matrix[ASC, I] za; // individual random effects (unscaled) (standardized component) for ASC
  matrix[I, (P-ASC)] z; // individual random effects (unscaled) (standardized component)
  vector<lower=0,upper=pi()/2>[P] tau_unif; // prior scale
  // matrix[ASC,ASC] L_Omega; // prior correlation
  vector[ASC] gamma; // ASC coeffs
}
transformed parameters {
  vector<lower=0>[P] tau;     // prior scale
  matrix[I, ASC] beta_ind;
  matrix[I, PN] betan_ind;
  
  for (k in 1:P) tau[k] = 2.5 * tan(tau_unif[k]);
  
  beta_ind = rep_matrix(gamma', I) + (diag_matrix(tau[1:ASC]) * za)';
  betan_ind = rep_matrix(betan', I) + z[,1:PN]*diag_matrix(tau[(ASC+1):(ASC+PN)]);
}

model {
  // create a temporary holding vector
  vector[I*T] log_prob; // we will add the log_prob for each scenario and pass it as a single vector to the posterior estimation engine
  vector[K] utils; // vector of utilities for each alternative
  int shift = 45; // integer shift between each iteration of j below (i.e., shift for number of attributes in each SP)
  int tshift; // total shift for each individual/task combination
  row_vector[K] ones = rep_row_vector(1, K);
  
  // priors on the parameters
  to_vector(za) ~ std_normal();
  to_vector(z) ~ std_normal();
  betan ~ normal(0, 5);
  to_vector(gamma) ~ normal(0, 5);
  
  // log probabilities of each choice in the dataset
  for(i in 1:I) {
    for(j in 1:T){
        tshift = shift*(j-1);
        utils[1] = X[i,(tshift+70)]*betan_ind[i,1] + X[i,(tshift+80)]*betan_ind[i,2] + X[i,(tshift+90)]*betan_ind[i,3] + (X[i,(tshift+65)]==1)*betan_ind[i,40];
        utils[2] = beta_ind[i,1] + X[i,(tshift+71)]*betan_ind[i,1] + X[i,5]*betan_ind[i,7] + X[i,12]*betan_ind[i,10] + X[i,17]*betan_ind[i,13] + (X[i,13]<40)*betan_ind[i,16] + (X[i,14]==1)*betan_ind[i,19] + (X[i,9]==1)*betan_ind[i,22] + X[i,10]*betan_ind[i,25] + X[i,16]*betan_ind[i,28] + X[i,30]*betan_ind[i,31] + X[i,29]*betan_ind[i,34] + X[i,31]*betan_ind[i,37] + (X[i,(tshift+66)]==3)*betan_ind[i,41] + (X[i,(tshift+66)]==4)*betan_ind[i,43] + (X[i,(tshift+66)]==5)*betan_ind[i,45];
        utils[3] = beta_ind[i,2] +                           + X[i,(tshift+87)]*betan_ind[i,4] + X[i,(tshift+97)]*betan_ind[i,5] + X[i,(tshift+77)]*betan_ind[i,6] + X[i,5]*betan_ind[i,8] + X[i,12]*betan_ind[i,11] + X[i,17]*betan_ind[i,14] + (X[i,13]<40)*betan_ind[i,17] + (X[i,14]==1)*betan_ind[i,20] + (X[i,9]==1)*betan_ind[i,23] + X[i,10]*betan_ind[i,26] + X[i,16]*betan_ind[i,29] + X[i,30]*betan_ind[i,32] + X[i,29]*betan_ind[i,35] + X[i,31]*betan_ind[i,38] + (X[i,(tshift+67)]==3)*betan_ind[i,42] + (X[i,(tshift+67)]==4)*betan_ind[i,44] + (X[i,(tshift+67)]==5)*betan_ind[i,46] + (X[i,(tshift+102)]==1.6)*betan_ind[i,47] + (X[i,(tshift+102)]==4.4)*betan_ind[i,49] + (X[i,(tshift+102)]==1)*betan_ind[i,51];
        utils[4] = beta_ind[i,3] + X[i,(tshift+73)]*betan_ind[i,1] + X[i,(tshift+89)]*betan_ind[i,4] + X[i,(tshift+99)]*betan_ind[i,5] + X[i,(tshift+79)]*betan_ind[i,6] + X[i,5]*betan_ind[i,9] + X[i,12]*betan_ind[i,12] + X[i,17]*betan_ind[i,15] + (X[i,13]<40)*betan_ind[i,18] + (X[i,14]==1)*betan_ind[i,21] + (X[i,9]==1)*betan_ind[i,24] + X[i,10]*betan_ind[i,27] + X[i,16]*betan_ind[i,30] + X[i,30]*betan_ind[i,33] + X[i,29]*betan_ind[i,36] + X[i,31]*betan_ind[i,39] + (X[i,(tshift+68)]==3)*betan_ind[i,41] + (X[i,(tshift+69)]==3)*betan_ind[i,42] + (X[i,(tshift+68)]==4)*betan_ind[i,43] + (X[i,(tshift+68)]==4)*betan_ind[i,44] + (X[i,(tshift+68)]==5)*betan_ind[i,45] + (X[i,(tshift+69)]==5)*betan_ind[i,46] + (X[i,(tshift+104)]==1.6)*betan_ind[i,48] + (X[i,(tshift+104)]==4.4)*betan_ind[i,50] + (X[i,(tshift+104)]==1)*betan_ind[i,52];
        log_prob[i+(j-1)*I] = ones * (log_softmax(utils) .* choice[i,(1+(j-1)):(K+(j-1))]');
    }
  }
  
  target += log_prob';
}
