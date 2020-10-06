function [theta,LLtab]=maxLL(Y, theta,sigma2,X0,epsilon)
%
% Conjugate gradient method for MLE computation
% 
%
%
%
%
% 


theta = theta(:);
nT = length(Y);
nX = length(X0);
nbiter = 100;

p =length(theta);

% Initialisation of the gradient method
[LL,resgradLL,reshessLL]=kalmangradient(Y,theta,sigma2,X0) ;

% Keep the likelihood values
LLtab=[];
LLtab=[LLtab,-LL];
% Current value of likelihood gradient
gradLL=-resgradLL(:);
% Current value of likelihood hessian
hessLL=-reshape(reshessLL,p,p);


u = gradLL;

theta_old = theta;
theta=theta-(gradLL'*u)/(u'*hessLL*u)*u;
Ngradp=norm(gradLL);


n=0;

while  (norm(gradLL) > epsilon)&&(n<nbiter)&&(norm(theta-theta_old) > epsilon) ,

    [LL,resgradLL,reshessLL]=kalmangradient(Y,theta,sigma2,X0) ;
    LLtab=[LLtab,-LL];

    % Current value of likelihood gradient
    gradLL=-resgradLL(:);
    % Current value of likelihood hessian
    hessLL=-reshape(reshessLL,p,p);

    % Descent of the gradient method
    u=gradLL+(norm(gradLL)/Ngradp)*u;


    theta_old = theta;
    % Current value of parameters
    theta = theta -(gradLL'*u)/(u'*hessLL*u)*u;

    n=n+1;
    Ngradp=norm(gradLL);

end;





