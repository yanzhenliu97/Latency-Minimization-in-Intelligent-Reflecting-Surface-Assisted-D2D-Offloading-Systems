function [result] = computeSingleDelay(weight,L,fi,fj,C,B,h,hi,hj,fai,sigma)
delay1=weight*L*C/fi;
rate=B*log2(1+abs(h+(conj(hi).*hj)'*fai)^2/sigma^2);
k=fi+fi^2/fj+fi^2/(C*rate);
delay2=weight*L*C/k;
result=delay1-delay2;
end