data {
  int T; // number of choice tasks
  int I; // number of Individuals / number of rows
  int C; // number of columns in data matrix
  int K; // number of alternatives
  int PN; // number of random covariates with normal priors
  int PLN; // number of random covariates with lognormal priors
  int PNLN; // number of random covariates with negative lognormal priors
  int P; // number of total random covariates
  matrix[I,(T*K)] choice; // binary indicator for choice
  matrix[I, C] X; // data matrix
}

// Following: https://mc-stan.org/docs/2_23/stan-users-guide/multivariate-hierarchical-priors-section.html
// for the multivariate component
parameters {
  vector[PN] betan; // hypermeans of the part-worths for random parameters with normal priors (less ASC)
  vector<lower=0>[PLN] betaln; // hypermeans of the part-worths for random parameters with postivie lognormal priors
  vector<lower=0>[PNLN] betanln; // hypermeans of the part-worths for random parameters with negative lognormal priors
  matrix[I, P] z; // individual random effects (unscaled) (standardized component)
  vector<lower=0,upper=pi()/2>[P] tau_unif; // prior scale
}
transformed parameters {
  vector<lower=0>[P] tau;     // prior scale
  matrix[I, PN] betan_ind;
  matrix[I, PLN] betaln_ind;
  matrix[I, PNLN] betanln_ind;
  
  for (k in 1:P) tau[k] = 2.5 * tan(tau_unif[k]);
  
  betan_ind = rep_matrix(betan', I) + z[,1:PN]*diag_matrix(tau[1:PN]);
  betaln_ind = rep_matrix(betaln', I) + z[,(PN+1):(PN+PLN)]*diag_matrix(tau[(PN+1):(PN+PLN)]);
  betanln_ind = rep_matrix(betanln', I) + z[,(PN+PLN+1):(PN+PLN+PNLN)]*diag_matrix(tau[(PN+PLN+1):(PN+PLN+PNLN)]);
}

model {
  // create a temporary holding vector
  vector[I*T] log_prob; // we will add the log_prob for each scenario and pass it as a single vector to the posterior estimation engine
  vector[K] utils; // vector of utilities for each alternative
  int shift = 45; // integer shift between each iteration of j below (i.e., shift for number of attributes in each SP)
  int tshift; // total shift for each individual/task combination
  row_vector[K] ones = rep_row_vector(1, K);
  
  // priors on the parameters
  to_vector(z) ~ std_normal();
  betan ~ normal(0, 5);
  betaln ~ lognormal(0, 5);
  
  // log probabilities of each choice in the dataset
  // Model is: [ASC] + [AV level] + [Single costs in $1000] + [Monthly costs in $100] + [Cost per hr] + [cost per 100 km] + [mileage] + [traffic conditions - average for alt4]
  for(i in 1:I) {
    for(j in 1:T){
        tshift = shift*(j-1);
        utils[1] = betan_ind[i,1] + X[i,(tshift+65)]*betan_ind[i,5] + X[i,(tshift+70)]*betanln_ind[i,1]                   +X[i,(tshift+90)]*betanln_ind[i,2]                                                +X[i,(tshift+80)]*betanln_ind[i,4]                              +X[i,(tshift+105)]*betanln_ind[i,5];
        utils[2] = betan_ind[i,2] + X[i,(tshift+66)]*betan_ind[i,5] + X[i,(tshift+71)]*betanln_ind[i,1]                   +X[i,(tshift+91)]*betanln_ind[i,2]                                                +X[i,(tshift+81)]*betanln_ind[i,4]+X[i,(tshift+102)]*betaln_ind[i,1]+X[i,(tshift+106)]*betanln_ind[i,5];
        utils[3] = betan_ind[i,3] + X[i,(tshift+67)]*betan_ind[i,5] + X[i,(tshift+87)]*betanln_ind[i,1]                   +X[i,(tshift+87)]*betanln_ind[i,2]                   +X[i,(tshift+77)]*betanln_ind[i,3]                                                           +X[i,(tshift+107)]*betanln_ind[i,5];
        utils[4] = betan_ind[i,4] + X[i,(tshift+68)]*betan_ind[i,5] + (X[i,(tshift+73)]+X[i,(tshift+89)])*betanln_ind[i,1]+(X[i,(tshift+89)]+X[i,(tshift+93)])*betanln_ind[i,2]+X[i,(tshift+79)]*betanln_ind[i,3]+X[i,(tshift+83)]*betanln_ind[i,4]+X[i,(tshift+104)]*betaln_ind[i,1]+(X[i,(tshift+108)]+X[i,(tshift+108)])/2*betanln_ind[i,5];
        log_prob[i+(j-1)*I] = ones * (log_softmax(utils) .* choice[i,(1+(j-1)):(K+(j-1))]');
    }
  }
  
  target += log_prob';
  target += -1*lognormal_lpdf(betanln | 0, 5);
}

