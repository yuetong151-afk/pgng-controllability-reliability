// M3: Baseline + lapse rate (based on Zorowitz M7)
// Same as M1 but adds lapse rate with informative prior (centred at ~0.12)

data {

    int<lower=1>  N;
    array[N] int<lower=1>  J;
    array[N] int<lower=1>  K;
    array[N] int<lower=0, upper=1>  Y;
    array[N] int<lower=0, upper=1>  R;
    array[N] int<lower=0, upper=1>  V;

}
transformed data {

    int  NJ = max(J);
    int  NK = max(K);

}
parameters {

    vector[7]     theta_mu;
    matrix[7,NJ]  theta_pr;
    vector<lower=0>[7] sigma;

}
transformed parameters {

    vector[NJ]  b1;
    vector[NJ]  b2;
    vector[NJ]  b3;
    vector[NJ]  b4;
    vector[NJ]  a1;
    vector[NJ]  a2;
    vector[NJ]  c1;

    {
    matrix[NJ,7] theta = transpose(diag_pre_multiply(sigma, theta_pr));

    b1 = (theta_mu[1] + theta[,1]) * 10;
    b2 = (theta_mu[2] + theta[,2]) * 10;
    b3 = (theta_mu[3] + theta[,3]) * 5;
    b4 = (theta_mu[4] + theta[,4]) * 5;
    a1 = Phi_approx(theta_mu[5] + theta[,5]);
    a2 = Phi_approx(theta_mu[6] + theta[,6]);
    c1 = Phi_approx(-2.0 + theta_mu[7] + 0.5 * theta[,7]);
    }

}
model {

    array[NJ, NK, 2] real Q = rep_array(0.5, NJ, NK, 2);
    vector[N] mu;

    for (n in 1:N) {
        real beta = (V[n] == 1) ? b1[J[n]] : b2[J[n]];
        real tau  = (V[n] == 1) ? b3[J[n]] : b4[J[n]];
        real eta  = (V[n] == 1) ? a1[J[n]] : a2[J[n]];
        real xi   = c1[J[n]];

        mu[n] = (0.5 * xi) + (1 - xi) * inv_logit(
            beta * (Q[J[n],K[n],2] - Q[J[n],K[n],1]) + tau
        );

        real delta = R[n] - Q[J[n],K[n],Y[n]+1];
        Q[J[n],K[n],Y[n]+1] += eta * delta;
    }

    target += bernoulli_lpmf(Y | mu);
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
    real  c1_mu = Phi_approx(-2.0 + theta_mu[7]);

}
