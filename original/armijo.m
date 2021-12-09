function [result]=armijo(theta,gradient,B,h,H,sigma,weight,L,k,a)
kesai=0.2;
alpha=2;
fai=exp(1j*theta);
max_iteration=30;

x=theta-gradient*alpha;
if compute_delay(B,h,H,x,sigma,weight,L,k,a)<=(compute_delay(B,h,H,theta,sigma,weight,L,k,a)-alpha*kesai*gradient'*gradient)
    i=1;
    while i<=max_iteration
        x=theta-alpha^(i+1)*gradient;
        if compute_delay(B,h,H,x,sigma,weight,L,k,a)>=(compute_delay(B,h,H,theta,sigma,weight,L,k,a)-alpha^(i+1)*kesai*gradient'*gradient)
            x=theta-alpha^(i)*gradient;
            break;
        end
        i=i+1;
    end
    
else
    i=1;
    while i<=max_iteration
        x=theta-alpha^(-i+1)*gradient;
        if compute_delay(B,h,H,x,sigma,weight,L,k,a)>=(compute_delay(B,h,H,theta,sigma,weight,L,k,a)-alpha^(-i+1)*kesai*gradient'*gradient)
        
            break;
        end
        i=i+1;
    end
   
    
end

result=x;
end