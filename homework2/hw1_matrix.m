u = [1; 2; 2];
sigma = [1, 0, 0; 0, 5, 2; 0, 2, 5];
x0 = [0.5; 0; 1];

% Calculate the determinant of the covariance matrix
detS = det(sigma);

% Dimensionality
d = 3;

% Compute the normalization constant
norm = 1 / ((2 * pi)^(d / 2) * sqrt(detS));

% Compute the exponent term
f = norm * exp(-0.5 * (x0 - u)' * inv(sigma) * (x0 - u))
[V, D] = eig(sigma);
V
D