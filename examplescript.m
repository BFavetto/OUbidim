
% Example of use for the Kalman-gradient algorithm
% B. Favetto and A. Samson, 2008
%
%
%%%%%%% AIM %%%%%%%
% Simulate hidden data from the Markov Chain, and a sample of observed data
% Compute the maximum likelihood estimator with conjugate gradient method
%
%
% The model is
%
% X(i) = A X(i-1) + eta_i
% y_i = H X(i) + sigma epsilon_i
%
% where 
% (X(i)) is a bidimensional Markov Chain
% A is a diagonal bidimensional matrix
% (eta_i) are independant Gaussian noise with a null mean and variance matrix Q
% (y_0, ..., y_n) are the discrete, partial and noisy observations of
% (X(i))
% H = (1 1)
% sigma is assumed to be known
% (epsilon_i) are independant Gaussian noise with a null mean and variance
% sigma2


% The unknown parameters are the two diagonal elements of A (theta_1, theta_2) and
% the three elements of Q (theta_3, theta_4, theta_5).


%%%%%%% Simulation of data %%%%%%%
%
% Simulation values for the parameters of the hidden Markov chain
H=[1 1];
A=[0.3 , 0 ; 0 , 0.8] ;
Q=[0.5 , 0.1 ; 0.1 , 1];
theta0= [A(1,1) , A(2,2) , Q(1,1) , Q(2,2) , Q(1,2)];
% variance of the observation noise 
sigma2 = 1;
% time
nT = 1000;
T = [1:nT];

% Initial state of the Markov chain
X0=[0;0];

% simulation of observations
[Y,X] = observ(nT,A,Q,H,sigma2,X0);
plot(T,Y,'*')
xlabel('time')
ylabel('observations')



%%%%%%%  Kalman gradient method %%%%%%%%%%%
%
% Initial values for the conjugate gradient method
thetastart=[0.2 , 0.9 , 0.6 , 0.9 , 0.15];
% Precision of the conjugate gradient method
epsilon = 0.01;
%
% Kalman gradient algorithm
[T,LL]=maxLL(Y,thetastart,sigma2,X0,epsilon);



%%%%%%%  EM method %%%%%%%%%%%
niter = 100;
theta = EM(Y,thetastart,sigma2,X0,niter,epsilon,H);
