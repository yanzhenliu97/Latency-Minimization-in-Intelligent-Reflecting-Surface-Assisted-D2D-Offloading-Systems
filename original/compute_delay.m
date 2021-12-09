function [result]=compute_delay(B,h,H,theta,sigma,weight,L,k,a)
fai=exp(1j*theta);
rate=compute_rate(B,h,H,fai,sigma);
delay=weight.*L./(k+rate)./a;
result=sum(delay);
end