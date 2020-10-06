% script de creation des donnees

% theta0
theta0= [0.3 , 0.8 , 0.5 , 1 , 0.1];
sigma2 = 0.2;
% temps
nT = 5000;

T = [1:nT];

A=[theta0(1) , 0 ; 0 , theta0(2)] ;
Q=[theta0(3) , theta0(5) ; theta0(5) , theta0(4)];
H=[1 1];
R=sigma2;
X0=[0;0];

for i=1:10

% observations bruitees
[Y,X] = observ(nT,A,Q,H,R,X0);

save(['data2D_5000_02_' int2str(i) '.mat'],'Y','X');

end;