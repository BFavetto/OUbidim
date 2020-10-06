function [Y,X]=observ(nT,A,Q,H,R,X0)
% Hidden Markov chain observed with additive noise
% 
% Model :
% X(n+1) = A.X(n) + Gaussian(0,Q)
% Y(n) = H.X(n) + Gaussian(0,R)
% nT = time ( process observed on [0,nT] )
%
%
% Input:
% X0 = initial position (0 if not defined)
% A = nX x nX matrix
% H = nY x nX matrix
% Q = nX x nX covariance matrix
% R = nY x nY covariance matrix
%
% Result :
% Y = nY x nT vector
% X = nX x nT vector





[nY,nX]=size(H);

if nargin<6, X0 = zeros(nX,1); end;

% calcul des ecarts types
S = sqrtm(Q);
D = sqrtm(R);

% initialisation
X = zeros(nX,nT);
X(:,1) = X0;
Y = zeros(nY,nT);
Y(:,1) = [H*X0+D*randn(nY , 1 )] ;

for i=2:(nT)
    %% etape de la chaine de markov
    X(:,i) = A*X(:, i - 1 ) + S*randn(nX,1) ;
    % observations bruitees
    Y(:,i) =  H*X(:, i) + D*randn(nY,1);
end;

