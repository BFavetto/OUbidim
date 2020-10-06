function [LLm,resgradLL,reshessLL]=kalmangradient(Y,theta,sigma2,X0)

% Computation of Kalman filter and its first and second order derivatives 
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


% initialisation
[nY,nT] = size(Y);
nX = length(X0);
p = length(theta);

Xf = zeros(nX,nT);
Xp = zeros(nX,nT);
Pf = zeros(nX,nX,nT);
Pp = zeros(nX,nX,nT);


W = zeros(1,nT);
InverseV = zeros(1,nT);
V = zeros(1,nT);

%Kalman filter initialisation
Xp(:,1) = X0; 
H=[1 1];


gradXp = zeros(nX,nT,p);
gradPp = zeros(nX,nX,nT,p);
gradXf = zeros(nX,nT,p);
gradPf = zeros(nX,nX,nT,p);
diffV = zeros(p,nT);
diffW = zeros(p,nT);

q = p*p;
hessXp = zeros(nX,nT,q);
hessXf = zeros(nX,nT,q);
hessV = zeros(q,nT);
hessW = zeros(q,nT);
hessPp = zeros(nX,nX,nT,q);
hessPf = zeros(nX,nX,nT,q);

LL=0;
diffLL = zeros(p,1);
hessLL = zeros(q,1);


A = [theta(1) , 0 ; 0 , theta(2)] ;
diffA = zeros(p,nX,nX);
diffA(1,:,:) = [1 , 0; 0 , 0];
diffA(2,:,:) = [0 , 0; 0 , 1];

Q = [theta(3) , theta(5) ; theta(5) , theta(4)];
diffQ = zeros(p,nX,nX);
diffQ(3,:,:) = [1 , 0 ; 0 , 0];
diffQ(4,:,:) = [0 , 0 ; 0 , 1];
diffQ(5,:,:) = [0 , 1 ; 1 , 0];


% First iteration
i = 1;

% first order derivatives
diffV(:,i) = cell2mat(arrayfun(@(j)  H*reshape(gradPp(:,:,i,j),nX,nX)*H',1:p,'UniformOutput',false)); %size p x 1
diffW(:,i) = cell2mat(arrayfun(@(j)  -H*reshape(gradXp(:,i,j),nX,1),1:p,'UniformOutput',false)); %size p x 1

% Second order derivatives
hessV(:,i) = cell2mat(arrayfun(@(j)  H*reshape(hessPp(:,:,i,j),nX,nX)*H',1:q,'UniformOutput',false)); %size q x 1
hessW(:,i) = cell2mat(arrayfun(@(j)  -H*reshape(hessXp(:,i,j),nX,1),1:q,'UniformOutput',false)); %size q x 1

% Kalman filter
V(i) = H*Pp(:,:,i)*H'+sigma2;
InverseV(i) = inv(V(i));
K = Pp(:,:,i)*H'*InverseV(i); % kalman gain
W(i) = Y(:,i) - H*Xp(:,i) ;
Xf(:,i) = Xp(:,i) + K*W(i);
Pf(:,:,i) = (eye(nX) - K*H)*Pp(:,:,i);


% first order derivatives
gradXf(:,i,:) = gradXp(:,i,:) + reshape(cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,j),nX,nX)*H'*W(i)*InverseV(i),1:p,'UniformOutput',false)) ...
    + cell2mat(arrayfun(@(j)  Pp(:,:,i)*H'*(diffW(j,i)*V(i) - W(i)*diffV(j,i))*InverseV(i)^2,1:p,'UniformOutput',false)),nX,1,p); %size p x nX x 1

gradPf(:,:,i,:) = reshape(cell2mat(arrayfun(@(j) (eye(nX) - K*H )*reshape(gradPp(:,:,i,j),nX,nX) - (reshape(gradPp(:,:,i,j),nX,nX)*(H'*H)*InverseV(i) ...
    - Pp(:,:,i)*(H'*H)*(diffV(j,i)*InverseV(i)^2))*Pp(:,:,i),1:p,'UniformOutput',false)),nX,nX,p);%size p x nX x 1

% Second order derivatives
hessXf(:,i,:) = hessXp(:,i,:) + reshape(cell2mat(arrayfun(@(j)  reshape(hessPp(:,:,i,j),nX,nX)*H'*W(i)*InverseV(i),1:q,'UniformOutput',false)) ...
    + cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*H'*(diffW(floor((j-1)/p)+1,i)*V(i)-W(i)*diffV(floor((j-1)/p)+1,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*H'*(diffW(rem(j-1,p)+1,i)*V(i)-W(i)*diffV(rem(j-1,p)+1,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*(hessW(j,i)*V(i)-W(i)*hessV(j,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*(-diffW(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)-diffW(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i) )*InverseV(i)^2,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*W(i)*diffV(rem(j-1,p)+1,i)*2*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3 ,1:q,'UniformOutput',false)), nX,1,q);

hessPf(:,:,i,:) = reshape(-cell2mat(arrayfun(@(j) (reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*(H'*H)*InverseV(i) ...
    - Pp(:,:,i)*(H'*H)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^2)*reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX),1:q,'UniformOutput',false))...
    - cell2mat(arrayfun(@(j)  (reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*(H'*H)*InverseV(i) - Pp(:,:,i)*(H'*H)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2)*reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX),1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j) (eye(nX) - Pp(:,:,i)*(H'*H)*InverseV(i))*reshape(hessPp(:,:,i,j),nX,nX) ,1:q,'UniformOutput',false))...
    - cell2mat(arrayfun(@(j) reshape(hessPp(:,:,i,j),nX,nX)*(H'*H)*InverseV(i)*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j) (reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*(H'*H)*diffV(rem(j-1,p)+1,i)  + reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*(H'*H)*diffV(floor((j-1)/p)+1,i))*InverseV(i)^2*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
    + cell2mat(arrayfun(@(j)  Pp(:,:,i)*(H'*H)*hessV(j,i)*InverseV(i)^2*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
    - cell2mat(arrayfun(@(j)   Pp(:,:,i)*(H'*H)*2*diffV(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3*Pp(:,:,i),1:q,'UniformOutput',false)),nX,nX,q);

% Liklihood
LL = LL -1/2*log(V(i)) - 1/2*(W(i)^2)*InverseV(i) ;
% Gradient of the likelihood
diffLL = diffLL -1/2*InverseV(i)*diffV(:,i) -1/2*(2*W(i)*InverseV(i)*diffW(:,i) + W(i)^2*(-InverseV(i)^2*diffV(:,i))) ;
% Hessian of the likelihood
hessLL = hessLL - 1/2*hessV(:,i)*InverseV(i) + 1/2*W(i)^2*hessV(:,i)*InverseV(i)^2 - W(i)*InverseV(i)*hessW(:,i) ...
    + cell2mat(arrayfun(@(j) 1/2*diffV(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2 ,1:q,'UniformOutput',false))'...
    - cell2mat(arrayfun(@(j) (diffW(floor((j-1)/p)+1,i)*diffW(rem(j-1,p)+1,i)*InverseV(i)-W(i)*diffW(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^2-W(i)*diffW(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2),1:q,'UniformOutput',false))'...
    - cell2mat(arrayfun(@(j) W(i)^2*diffV(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3,1:q,'UniformOutput',false))';



for i=2:nT

    % prediction
    Xp(:,i) = A*Xf(:,i-1) ;
    Pp(:,:,i) = A*Pf(:,:,i-1)*A' + Q;

    % first order derivatives
    gradXp(:,i,:) = reshape(cell2mat(arrayfun(@(j)  reshape(diffA(j,:,:),nX,nX)*Xf(:,i-1) + A*reshape(gradXf(:,i-1,j),nX,1) ,1:p,'UniformOutput',false)),nX,1,p); %size p x nX x 1
    gradPp(:,:,i,:) = reshape(cell2mat(arrayfun(@(j) reshape(diffA(j,:,:),nX,nX)*Pf(:,:,i-1)*A' + A*reshape(gradPf(:,:,i-1,j),nX,nX)*A' ...
        + A*Pf(:,:,i-1)*reshape(diffA(j,:,:),nX,nX)' + reshape(diffQ(j,:,:),nX,nX) ,1:p,'UniformOutput',false)),nX,nX,p); %size p x nX x nX x 1


    % Second order derivatives
    hessXp(:,i,:) = reshape(A*reshape(hessXf(:,i-1,:),nX,q) + cell2mat(arrayfun(@(j) reshape(diffA(floor((j-1)/p)+1,:,:),nX,nX)*reshape(gradXf(:,i-1,rem(j-1,p)+1),nX,1) ...
        + reshape(diffA(rem(j-1,p)+1,:,:),nX,nX)*reshape(gradXf(:,i-1,floor((j-1)/p)+1),nX,1) ,1:q,'UniformOutput',false)),nX,1,q);

    hessPp(:,:,i,:) = reshape(cell2mat(arrayfun(@(j) A*hessPf(:,:,i-1,j)*A',1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) reshape(diffA(floor((j-1)/p)+1,:,:),nX,nX)*reshape(gradPf(:,:,i-1,rem(j-1,p)+1),nX,nX)*A'+A*reshape(gradPf(:,:,i-1,floor((j-1)/p)+1),nX,nX)*reshape(diffA(rem(j-1,p)+1,:,:),nX,nX)',1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) reshape(diffA(floor((j-1)/p)+1,:,:),nX,nX)*Pf(:,:,i-1)*reshape(diffA(rem(j-1,p)+1,:,:),nX,nX)' + reshape(diffA(rem(j-1,p)+1,:,:),nX,nX)*Pf(:,:,i-1)*reshape(diffA(floor((j-1)/p)+1,:,:),nX,nX)',1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) reshape(diffA(rem(j-1,p)+1,:,:),nX,nX)*reshape(gradPf(:,:,i-1,floor((j-1)/p)+1),nX,nX)*A' + A*gradPf(:,:,i-1,rem(j-1,p)+1)*reshape(diffA(floor((j-1)/p)+1,:,:),nX,nX)' ,1:q,'UniformOutput',false)),nX,nX,1,q);


    % Intermediate computations
    V(i) = H*Pp(:,:,i)*H'+sigma2;
    InverseV(i) = inv(V(i));
    K = Pp(:,:,i)*H'*InverseV(i); % kalman gain
    W(i) = Y(:,i) - H*Xp(:,i) ;

    % first order derivatives
    diffV(:,i) = cell2mat(arrayfun(@(j)  H*reshape(gradPp(:,:,i,j),nX,nX)*H',1:p,'UniformOutput',false)); %size p x 1
    diffW(:,i) = cell2mat(arrayfun(@(j)  -H*reshape(gradXp(:,i,j),nX,1),1:p,'UniformOutput',false)); %size p x 1

    % Second order derivatives
    hessV(:,i) = cell2mat(arrayfun(@(j)  H*reshape(hessPp(:,:,i,j),nX,nX)*H',1:q,'UniformOutput',false)); %size q x 1
    hessW(:,i) = cell2mat(arrayfun(@(j)  -H*reshape(hessXp(:,i,j),nX,1),1:q,'UniformOutput',false)); %size q x 1

    % updating
    Xf(:,i) = Xp(:,i) + K*W(i);
    Pf(:,:,i) = (eye(nX) - K*H)*Pp(:,:,i);

    % first order derivatives
    gradXf(:,i,:) = gradXp(:,i,:) + reshape(cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,j),nX,nX)*H'*W(i)*InverseV(i),1:p,'UniformOutput',false)) ...
        + cell2mat(arrayfun(@(j)  Pp(:,:,i)*H'*(diffW(j,i)*V(i) - W(i)*diffV(j,i))*InverseV(i)^2,1:p,'UniformOutput',false)),nX,1,p); %size p x nX x 1

    gradPf(:,:,i,:) = reshape(cell2mat(arrayfun(@(j) (eye(nX) - K*H )*reshape(gradPp(:,:,i,j),nX,nX) - (reshape(gradPp(:,:,i,j),nX,nX)*(H'*H)*InverseV(i) ...
        - Pp(:,:,i)*(H'*H)*(diffV(j,i)*InverseV(i)^2))*Pp(:,:,i),1:p,'UniformOutput',false)),nX,nX,p);%size p x nX x 1

    % Second order derivatives
    hessXf(:,i,:) = hessXp(:,i,:) + reshape(cell2mat(arrayfun(@(j)  reshape(hessPp(:,:,i,j),nX,nX)*H'*W(i)*InverseV(i),1:q,'UniformOutput',false)) ...
        + cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*H'*(diffW(floor((j-1)/p)+1,i)*V(i)-W(i)*diffV(floor((j-1)/p)+1,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j)  reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*H'*(diffW(rem(j-1,p)+1,i)*V(i)-W(i)*diffV(rem(j-1,p)+1,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*(hessW(j,i)*V(i)-W(i)*hessV(j,i))*InverseV(i)^2,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*(-diffW(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)-diffW(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i) )*InverseV(i)^2,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) Pp(:,:,i)*H'*W(i)*diffV(rem(j-1,p)+1,i)*2*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3 ,1:q,'UniformOutput',false)), nX,1,q);


    hessPf(:,:,i,:) = reshape(-cell2mat(arrayfun(@(j) (reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*(H'*H)*InverseV(i) ...
        - Pp(:,:,i)*(H'*H)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^2)*reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX),1:q,'UniformOutput',false))...
        - cell2mat(arrayfun(@(j)  (reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*(H'*H)*InverseV(i) - Pp(:,:,i)*(H'*H)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2)*reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX),1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) (eye(nX) - Pp(:,:,i)*(H'*H)*InverseV(i))*reshape(hessPp(:,:,i,j),nX,nX) ,1:q,'UniformOutput',false))...
        - cell2mat(arrayfun(@(j) reshape(hessPp(:,:,i,j),nX,nX)*(H'*H)*InverseV(i)*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j) (reshape(gradPp(:,:,i,floor((j-1)/p)+1),nX,nX)*(H'*H)*diffV(rem(j-1,p)+1,i)  + reshape(gradPp(:,:,i,rem(j-1,p)+1),nX,nX)*(H'*H)*diffV(floor((j-1)/p)+1,i))*InverseV(i)^2*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
        + cell2mat(arrayfun(@(j)  Pp(:,:,i)*(H'*H)*hessV(j,i)*InverseV(i)^2*Pp(:,:,i)  ,1:q,'UniformOutput',false))...
        - cell2mat(arrayfun(@(j)   Pp(:,:,i)*(H'*H)*2*diffV(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3*Pp(:,:,i),1:q,'UniformOutput',false)),nX,nX,q);

    % Likelihood
    LL = LL -1/2*log(V(i)) - 1/2*(W(i)^2)*InverseV(i) ;

    % Gradient of the likelihood
    diffLL = diffLL -1/2*InverseV(i)*diffV(:,i) -1/2*(2*W(i)*InverseV(i)*diffW(:,i) + W(i)^2*(-InverseV(i)^2*diffV(:,i))) ;
    
    % Hessian of the likelihood

    hessLL = hessLL - 1/2*hessV(:,i)*InverseV(i) + 1/2*W(i)^2*hessV(:,i)*InverseV(i)^2 - W(i)*InverseV(i)*hessW(:,i) ...
        + cell2mat(arrayfun(@(j) 1/2*diffV(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2 ,1:q,'UniformOutput',false))'...
        - cell2mat(arrayfun(@(j) (diffW(floor((j-1)/p)+1,i)*diffW(rem(j-1,p)+1,i)*InverseV(i)-W(i)*diffW(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^2-W(i)*diffW(floor((j-1)/p)+1,i)*diffV(rem(j-1,p)+1,i)*InverseV(i)^2),1:q,'UniformOutput',false))'...
        - cell2mat(arrayfun(@(j) W(i)^2*diffV(rem(j-1,p)+1,i)*diffV(floor((j-1)/p)+1,i)*InverseV(i)^3,1:q,'UniformOutput',false))';

end;



resgradLL = -1000000*diffLL;

reshessLL = -1000000*hessLL;

LLm=-1000000*LL;



