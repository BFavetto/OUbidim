function [Y,X]=simul(nT,A,B,Q,H,R,X0)

% Hidden Markov chain observed with additive noise
% 
% Model :
% X(n) = A(n).X(n-1) + B(n) + Gaussian(0,Q(n))
% Y(n) = H(n).X(n) + Gaussian(0,R(n))
% nT = time ( process observed on [0,nT] )
%
%
% Input:
% X0 = initial position (0 if not defined)
% A = nX x nX x nT matrix
% H = nY x nX matrix
% Q = nX x nX x nT covariance matrix
% R = nY x nY covariance matrix
%
% Result :
% Y = nY x nT vector
% X = nX x nT vector





[nY,nX]=size(H(:,:,1));

if nargin<7, X0 = zeros(nX,1); end;



% initialisation
X = zeros(nX,nT);
X(:,1) = X0;
Y = zeros(nY,nT);
Y(:,1) = [H(:,:,1)*X0+sqrtm(R(:,:,1))*randn(nY , 1 )] ;

for i=2:(nT)
    %% etape de la chaine de markov
    X(:,i) = A(:,:,i)*X(:, i - 1 ) + B(:,i) + sqrtm(Q(:,:,i))*randn(nX,1) ;
    % observations bruitees
    Y(:,i) =  H(:,:,i)*X(:, i) + sqrtm(R(:,:,i))*randn(nY,1);
end;