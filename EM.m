function  theta = EM(Y,theta0,sigma2,X0,niter,epsilon,J)

[nY,nT] = size(Y);
%true values
nX = 2;


A = [theta0(1) , 0 ; 0 ,theta0(2)];
Q = [theta0(3) , theta0(5); theta0(5) , theta0(4) ];

A = repmat(A,[1 1 nT]);
B = repmat([0; 0],1,nT);
H = repmat([1 1],[1 1 nT]);
C = zeros(1,nT);
Q = repmat(Q,[1 1 nT]);
R = repmat(sigma2,[1 1 nT]);


%keep the estimates
theta = [ A(1,1,2) A(2,2,2) Q(1,1,2) Q(2,2,2) Q(1,2,2)];

iter = 1;
kalman_smoother(Y,R,Q,A,H,B,C,X0)

global Xs Ps Cs

%Maximisation step
s = (Y(1) -H(:,:,1)*Xs(:,1)).^2 + H(:,:,1)*Ps(:,:,1)*H(:,:,1)';
M1 = zeros(nX,nX);
M2 = zeros(nX,nX);
V = zeros(nX,nX);

for i=2:nT
    s = s + (Y(i) - H(:,:,1)*Xs(:,i)).^2 + H(:,:,1)*Ps(:,:,i)*H(:,:,1)';
    M2 = M2 + Ps(:,:,i) + Xs(:,i)*Xs(:,i)';
    V = V + Ps(:,:,i-1) + Xs(:,i-1)*Xs(:,i-1)';
    M1 = M1 + Cs(:,:,i)+ Xs(:,i)*Xs(:,i-1)';
end


A = diag(diag(M1*inv(V)));
Q = (M2-M1*inv(V)*M1')/(nT-1);

Q = repmat(Q, [1 1 nT]);
A = repmat(A, [1 1 nT]);


%Keep the estimates
theta = [theta; A(1,1,2) A(2,2,2) Q(1,1,2) Q(2,2,2) Q(1,2,2)];

iter = 2;
%for i=1:niter-1
while (norm(theta(iter,:)-theta(iter-1,:)) > epsilon)&&(iter<niter)

    %Expectation step: computation of Kalman smoother
    kalman_smoother(Y,R,Q,A,H,B,C,X0)

    global Xs Ps Cs

    %Maximisation step
    s = (Y(1) -H(:,:,1)*Xs(:,1)).^2 + H(:,:,1)*Ps(:,:,1)*H(:,:,1)';
    M1 = zeros(nX,nX);
    M2 = zeros(nX,nX);
    V = zeros(nX,nX);

    for i=2:nT
        s = s + (Y(i) - H(:,:,1)*Xs(:,i)).^2 + H(:,:,1)*Ps(:,:,i)*H(:,:,1)';
        M2 = M2 + Ps(:,:,i) + Xs(:,i)*Xs(:,i)';
        V = V + Ps(:,:,i-1) + Xs(:,i-1)*Xs(:,i-1)';
        M1 = M1 + Cs(:,:,i)+ Xs(:,i)*Xs(:,i-1)';
    end

    A = diag(diag(M1*inv(V)));
    Q = (M2-M1*inv(V)*M1')/(nT-1);

    Q = repmat(Q, [1 1 nT]);
    A = repmat(A, [1 1 nT]);


    %Keep the estimates
    theta = [theta; A(1,1,2) A(2,2,2) Q(1,1,2) Q(2,2,2) Q(1,2,2)];

    iter = iter+1;
end


