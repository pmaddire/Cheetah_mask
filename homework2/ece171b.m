% Define state-space matrices
A = [1, 2, 0, 0;
     0, 1, 0, 0;
     0, 0, 0, 1;
     0, 0, 0, -1];

B = [1;
     0;
     0;
     1];

C = [1, 0, 1, 0];

D = 0;

% Create the state-space system
sys = ss(A, B, C, D);

% Compute the minimal realization of the system
[sysr, T] = minreal(sys)