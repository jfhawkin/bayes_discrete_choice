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
  vector<lower=0,upper=pi()/2>[P] tau_unif; // prior scale
  // matrix[ASC,ASC] L_Omega; // prior correlation
  vector[ASC] gamma; // ASC coeffs
}
transformed parameters {
  vector<lower=0>[ASC] tau;     // prior scale
  matrix[I, ASC] beta_ind;
  
  for (k in 1:ASC) tau[k] = 2.5 * tan(tau_unif[k]);
  
  beta_ind = rep_matrix(gamma', I) + (diag_matrix(tau) * za)';
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
  betan ~ normal(0, 5);
  // L_Omega ~ lkj_corr_cholesky(2);
  to_vector(gamma) ~ normal(0, 5);
  
  // log probabilities of each choice in the dataset
  for(i in 1:I) {
    for(j in 1:T){
        tshift = shift*(j-1);
        utils[1] = X[i,(tshift+70)]*betan[1] + X[i,(tshift+80)]*betan[2] + X[i,(tshift+90)]*betan[3] + (X[i,(tshift+65)]==1)*betan[40];
        utils[2] = beta_ind[i,1] + X[i,(tshift+71)]*betan[1] + X[i,5]*betan[7] + X[i,12]*betan[10] + X[i,17]*betan[13] + (X[i,13]<40)*betan[16] + (X[i,14]==1)*betan[19] + (X[i,9]==1)*betan[22] + X[i,10]*betan[25] + X[i,16]*betan[28] + X[i,30]*betan[31] + X[i,29]*betan[34] + X[i,31]*betan[37] + (X[i,(tshift+66)]==3)*betan[41] + (X[i,(tshift+66)]==4)*betan[43] + (X[i,(tshift+66)]==5)*betan[45];
        utils[3] = beta_ind[i,2] +                           + X[i,(tshift+87)]*betan[4] + X[i,(tshift+97)]*betan[5] + X[i,(tshift+77)]*betan[6] + X[i,5]*betan[8] + X[i,12]*betan[11] + X[i,17]*betan[14] + (X[i,13]<40)*betan[17] + (X[i,14]==1)*betan[20] + (X[i,9]==1)*betan[23] + X[i,10]*betan[26] + X[i,16]*betan[29] + X[i,30]*betan[32] + X[i,29]*betan[35] + X[i,31]*betan[38] + (X[i,(tshift+67)]==3)*betan[42] + (X[i,(tshift+67)]==4)*betan[44] + (X[i,(tshift+67)]==5)*betan[46] + (X[i,(tshift+102)]==1.6)*betan[47] + (X[i,(tshift+102)]==4.4)*betan[49] + (X[i,(tshift+102)]==1)*betan[51];
        utils[4] = beta_ind[i,3] + X[i,(tshift+73)]*betan[1] + X[i,(tshift+89)]*betan[4] + X[i,(tshift+99)]*betan[5] + X[i,(tshift+79)]*betan[6] + X[i,5]*betan[9] + X[i,12]*betan[12] + X[i,17]*betan[15] + (X[i,13]<40)*betan[18] + (X[i,14]==1)*betan[21] + (X[i,9]==1)*betan[24] + X[i,10]*betan[27] + X[i,16]*betan[30] + X[i,30]*betan[33] + X[i,29]*betan[36] + X[i,31]*betan[39] + (X[i,(tshift+68)]==3)*betan[41] + (X[i,(tshift+69)]==3)*betan[42] + (X[i,(tshift+68)]==4)*betan[43] + (X[i,(tshift+68)]==4)*betan[44] + (X[i,(tshift+68)]==5)*betan[45] + (X[i,(tshift+69)]==5)*betan[46] + (X[i,(tshift+104)]==1.6)*betan[48] + (X[i,(tshift+104)]==4.4)*betan[50] + (X[i,(tshift+104)]==1)*betan[52];
        log_prob[i+(j-1)*I] = ones * (log_softmax(utils) .* choice[i,(1+(j-1)):(K+(j-1))]');
    }
  }
  
  target += log_prob';
}
