function kalman_smoother(Y,R,Q,A,H,B,C,X0)
% MODEL :
%
% X(n) = A(:,:,n).X(n-1) + B(:,n) + epsilon(n) -> hidden
% Y(n) = H(:,:,n).X(n) + C(:,n) + eta(n) -> observed
% with
% epsilon(n) ~ (i.i.d) Gaussian(0,Q(:,:,n))
% eta(n) ~ (i.i.d) Gaussian(0,R(:,:,n))
% initial value follow a Gaussian(X0,P0) distribution 
%
%
% X is a matrix of size nX x nT 
% Y is a matrix of size nY x nT
% A is a matrix of size nX x nX x nT
% B is a matrix of size nX x nT
% H is a matrix of size nY x nX x nT
% C is a matrix of size nY x nT
% Q is a matrix of size nX x nX x nT
% R is a matrix of size nY x nY x nT
% X0 is a vector nX x 1
% P0 is a matrix nX x nX
% P is transit matrix
%
% INPUT :
% Y : observation vector
% R : observation noise correlation matrix  
% Q : innovation noise correlation matrix
% A : innovation matrix
% H : 
% B : deterministic hidden drift
% C : deterministic observed drift
% X0 : known initial mean 
% P0 : known initial covariance matrix, default 0 (deterministic case)

global Inverse Determ Xf Xp Pf Pp DevY Xs Ps Us Uf Cs 


% initialisation
[nY,nT] = size(Y);
nX = length(X0);

%filter
Xf = zeros(nX,nT); 
Xp = zeros(nX,nT);
Pf = zeros(nX,nX,nT); 
Uf = zeros(nX,nT);
Pp = zeros(nX,nX,nT);
DevY = zeros(nY,nT);
Yp = zeros(nY,nT);
K = zeros(nX,nY,nT);

Inverse = zeros(nY,nY,nT);
Determ = zeros(1,nT);

%filter initialization
Xp(:,1) = X0; 
%if non deterministic initial value
if nargin==9, Pp(:,:,1)=P0; end 

%0 step
Determ(1) = det(H(:,:,1)*Pp(:,:,1)*H(:,:,1)'+R(:,:,1));
Inverse(:,:,1) = inv(H(:,:,1)*Pp(:,:,1)*H(:,:,1)'+R(:,:,1)); 
K(:,:,1) = Pp(:,:,1)*H(:,:,1)'*Inverse(:,:,1); % kalman gain
Yp(:,1) = H(:,:,1)*Xp(:,1) + C(:,1);
DevY(:,1) = Y(:,1) - Yp(:,1);

Xf(:,1) = Xp(:,1) + K(:,:,1)*DevY(:,1);
Pf(:,:,1) = (eye(nX) - K(:,:,1)*H(:,:,1))*Pp(:,:,1);
%Uf(:,1) = P*Xf(:,1);


for i=2:nT

    %% prediction
    Xp(:,i) = A(:,:,i)*Xf(:,i-1) + B(:,i);
    Pp(:,:,i) = A(:,:,i)*Pf(:,:,i-1)*A(:,:,i)' + Q(:,:,i);
   
    %% updating
    Determ(i) = det(H(:,:,i)*Pp(:,:,i)*H(:,:,i)'+R(:,:,i));
    Inverse(:,:,i) = inv(H(:,:,i)*Pp(:,:,i)*H(:,:,i)'+R(:,:,i));
    K(:,:,i) = Pp(:,:,i)*H(:,:,i)'*Inverse(:,:,i); % kalman gain
    Yp(:,i) = H(:,:,i)*Xp(:,i) + C(:,i);
    DevY(:,i) = Y(:,i) - Yp(:,i);
    Xf(:,i) = Xp(:,i) + K(:,:,i)*DevY(:,i);
    Pf(:,:,i) = (eye(nX) - K(:,:,i)*H(:,:,i))*Pp(:,:,i);
%    Uf(:,i) = P*Xf(:,i);
    
    F(:,:,i-1) = Pf(:,:,i-1)*A(:,:,i-1)'*inv(Pp(:,:,i));
    Cf(:,:,i) = (eye(nX) - K(:,:,i)*H(:,:,i))*A(:,:,i)*Pf(:,:,i-1);
   
end;

%smoother
Xs = zeros(nX,nT);
Us = zeros(nX,nT);
Ps = zeros(nX,nX,nT); 
Cs = zeros(nX,nX,nT);
Cs2 = zeros(nX,nX,nT);

%smoother initialization
Xs(:,end) = Xf(:,end);
%Us(:,end) = P*Xs(:,end);
Ps(:,:,end) = Pf(:,:,end);

for i=1:nT-1
    j = nT-i;

    Xs(:,j) = Xf(:,j) + F(:,:,j)*(Xs(:,j+1) - Xp(:,j+1));
  %  Us(:,j) = P*Xs(:,j);
    Ps(:,:,j) = Pf(:,:,j) + F(:,:,j)*(Ps(:,:,j+1) - Pp(:,:,j+1))*F(:,:,j)';
    
    Cs(:,:,j+1) = (eye(nX) + (Ps(:,:,j+1)-Pf(:,:,j+1))*inv(Pf(:,:,j+1)))*Cf(:,:,j+1);

end







