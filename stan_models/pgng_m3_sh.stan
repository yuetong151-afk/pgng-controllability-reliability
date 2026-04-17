// M3_sh: Baseline + lapse rate, split-half / test-retest version
// Adds M (session indicator) dimension to M3
// Parameters: b1, b2, b3, b4, a1, a2, c1 (7 params)
// theta_c_pr: common subject effect (shared across sessions)
// theta_d_pr: divergent subject effect (differs between sessions)
data {
    int<lower=1>  N;
    array[N] int<lower=1>  J;
    array[N] int<lower=1>  K;
    array[N] int<lower=1, upper=2>  M;
    array[N] int<lower=0, upper=1>  Y;
    array[N] int<lower=0, upper=1>  R;
    array[N] int<lower=0, upper=1>  V;
}
transformed data {
    int  NJ = max(J);
    int  NK = max(K);
}
parameters {
    matrix[7,2]   theta_mu;               // Population-level effects (7 params x 2 sessions)
    matrix[7,NJ]  theta_c_pr;             // Standardized subject-level effects (common)
    matrix[7,NJ]  theta_d_pr;             // Standardized subject-level effects (divergent)
    matrix<lower=0>[7,2] sigma;           // Subject-level standard deviations
}
transformed parameters {
    array[2] vector[NJ]  b1;
    array[2] vector[NJ]  b2;
    array[2] vector[NJ]  b3;
    array[2] vector[NJ]  b4;
    array[2] vector[NJ]  a1;
    array[2] vector[NJ]  a2;
    array[2] vector[NJ]  c1;
    {
    matrix[NJ,7] theta_c = transpose(diag_pre_multiply(sigma[,1], theta_c_pr));
    matrix[NJ,7] theta_d = transpose(diag_pre_multiply(sigma[,2], theta_d_pr));
    b1[1] = (theta_mu[1,1] + theta_c[,1] - theta_d[,1]) * 10;
    b1[2] = (theta_mu[1,2] + theta_c[,1] + theta_d[,1]) * 10;
    b2[1] = (theta_mu[2,1] + theta_c[,2] - theta_d[,2]) * 10;
    b2[2] = (theta_mu[2,2] + theta_c[,2] + theta_d[,2]) * 10;
    b3[1] = (theta_mu[3,1] + theta_c[,3] - theta_d[,3]) * 5;
    b3[2] = (theta_mu[3,2] + theta_c[,3] + theta_d[,3]) * 5;
    b4[1] = (theta_mu[4,1] + theta_c[,4] - theta_d[,4]) * 5;
    b4[2] = (theta_mu[4,2] + theta_c[,4] + theta_d[,4]) * 5;
    a1[1] = Phi_approx(theta_mu[5,1] + theta_c[,5] - theta_d[,5]);
    a1[2] = Phi_approx(theta_mu[5,2] + theta_c[,5] + theta_d[,5]);
    a2[1] = Phi_approx(theta_mu[6,1] + theta_c[,6] - theta_d[,6]);
    a2[2] = Phi_approx(theta_mu[6,2] + theta_c[,6] + theta_d[,6]);
    c1[1] = Phi_approx(-2.0 + theta_mu[7,1] + 0.5 * (theta_c[,7] - theta_d[,7]));
    c1[2] = Phi_approx(-2.0 + theta_mu[7,2] + 0.5 * (theta_c[,7] + theta_d[,7]));
    }
}
model {
    array[NJ, NK, 2, 2] real Q;
    for (j in 1:NJ) for (k in 1:NK) for (m in 1:2) for (a in 1:2) Q[j,k,m,a] = 0.5;
    for (n in 1:N) {
        real beta = (V[n] == 1) ? b1[M[n],J[n]] : b2[M[n],J[n]];
        real tau  = (V[n] == 1) ? b3[M[n],J[n]] : b4[M[n],J[n]];
        real eta  = (V[n] == 1) ? a1[M[n],J[n]] : a2[M[n],J[n]];
        real xi   = c1[M[n],J[n]];
        real mu   = (0.5 * xi) + (1 - xi) * inv_logit(
            beta * (Q[J[n],K[n],M[n],2] - Q[J[n],K[n],M[n],1]) + tau
        );
        target += bernoulli_lpmf(Y[n] | mu);
        real delta = R[n] - Q[J[n],K[n],M[n],Y[n]+1];
        Q[J[n],K[n],M[n],Y[n]+1] += eta * delta;
    }
    target += std_normal_lpdf(to_vector(theta_mu));
    target += std_normal_lpdf(to_vector(theta_c_pr));
    target += std_normal_lpdf(to_vector(theta_d_pr));
    target += student_t_lpdf(to_vector(sigma) | 3, 0, 1);
}
generated quantities {
    // Group-level means per session
    real b1_mu_s1 = theta_mu[1,1] * 10;
    real b1_mu_s2 = theta_mu[1,2] * 10;
    real b2_mu_s1 = theta_mu[2,1] * 10;
    real b2_mu_s2 = theta_mu[2,2] * 10;
    real b3_mu_s1 = theta_mu[3,1] * 5;
    real b3_mu_s2 = theta_mu[3,2] * 5;
    real b4_mu_s1 = theta_mu[4,1] * 5;
    real b4_mu_s2 = theta_mu[4,2] * 5;
    real a1_mu_s1 = Phi_approx(theta_mu[5,1]);
    real a1_mu_s2 = Phi_approx(theta_mu[5,2]);
    real a2_mu_s1 = Phi_approx(theta_mu[6,1]);
    real a2_mu_s2 = Phi_approx(theta_mu[6,2]);
    real c1_mu_s1 = Phi_approx(-2.0 + theta_mu[7,1]);
    real c1_mu_s2 = Phi_approx(-2.0 + theta_mu[7,2]);
}
