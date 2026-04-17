data {

    // Metadata
    int<lower=1>  N;                        // Number of total observations
    array[N] int<lower=1>  J;               // Subject-indicator per observation
    array[N] int<lower=1>  K;               // Bandit-indicator per observation

    // Data
    array[N] int<lower=0, upper=1>  Y;      // Response (go = 1, no-go = 0)
    array[N] int<lower=0, upper=1>  R;      // Outcome (better = 1, worse = 0)
    array[N] int<lower=0, upper=1>  V;      // Valence (positive = 1, negative = 0)
    array[N] int<lower=0, upper=1>  C;      // Controllability (controllable = 1, uncontrollable = 0)

}
transformed data {

    int  NJ = max(J);                       // Number of total subjects
    int  NK = max(K);                       // Number of total bandits

}
parameters {

    // Participant parameters
    vector[8]     theta_mu;                 // Population-level effects (6 original + 2 delta)
    matrix[8,NJ]  theta_pr;                 // Standardized subject-level effects

    // Parameter variances
    vector<lower=0>[8] sigma;               // Subject-level standard deviations

}
transformed parameters {

    vector[NJ]  b1;                         // Inverse temperature (positive valence)
    vector[NJ]  b2;                         // Inverse temperature (negative valence)
    vector[NJ]  b3;                         // Go bias (positive valence)
    vector[NJ]  b4;                         // Go bias (negative valence)
    vector[NJ]  a1;                         // Learning rate (positive valence)
    vector[NJ]  a2;                         // Learning rate (negative valence)
    vector[NJ]  d1;                         // Controllability modulation (positive valence)
    vector[NJ]  d2;                         // Controllability modulation (negative valence)

    // Construction block
    {

    // Rotate random effects
    matrix[NJ,8] theta = transpose(diag_pre_multiply(sigma, theta_pr));

    // Construct random effects
    b1 = (theta_mu[1] + theta[,1]) * 10;
    b2 = (theta_mu[2] + theta[,2]) * 10;
    b3 = (theta_mu[3] + theta[,3]) * 5;
    b4 = (theta_mu[4] + theta[,4]) * 5;
    a1 = Phi_approx(theta_mu[5] + theta[,5]);
    a2 = Phi_approx(theta_mu[6] + theta[,6]);
    d1 = (theta_mu[7] + theta[,7]) * 5;    // delta+ (reward domain)
    d2 = (theta_mu[8] + theta[,8]) * 5;    // delta- (punishment domain)

    }

}
model {

    // Initialize Q-values
    array[NJ, NK, 2] real Q = rep_array(0.5, NJ, NK, 2);

    // Construct linear predictor
    vector[N] mu;
    for (n in 1:N) {

        // Assign trial-level parameters
        real beta  = (V[n] == 1) ? b1[J[n]] : b2[J[n]];
        real tau   = (V[n] == 1) ? b3[J[n]] : b4[J[n]];
        real eta   = (V[n] == 1) ? a1[J[n]] : a2[J[n]];
        real delta_c = (V[n] == 1) ? d1[J[n]] : d2[J[n]];

        // Compute (scaled) difference in state-action values
        // + controllability modulation term (delta * Controllable)
        mu[n] = beta * (Q[J[n],K[n],2] - Q[J[n],K[n],1]) + tau + delta_c * C[n];

        // Compute prediction error
        real delta = R[n] - Q[J[n],K[n],Y[n]+1];

        // Update state-action values
        Q[J[n],K[n],Y[n]+1] += eta * delta;

    }

    // Likelihood
    target += bernoulli_logit_lpmf(Y | mu);

    // Priors
    target += std_normal_lpdf(theta_mu);
    target += std_normal_lpdf(to_vector(theta_pr));
    target += student_t_lpdf(sigma | 3, 0, 1);

}
generated quantities {

    real  b1_mu = theta_mu[1] * 10;
    real  b2_mu = theta_mu[2] * 10;
    real  b3_mu = theta_mu[3] * 5;
    real  b4_mu = theta_mu[4] * 5;
    real  a1_mu = Phi_approx(theta_mu[5]);
    real  a2_mu = Phi_approx(theta_mu[6]);
    real  d1_mu = theta_mu[7] * 5;         // delta+ group mean
    real  d2_mu = theta_mu[8] * 5;         // delta- group mean

}
