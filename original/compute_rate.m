function [result]=compute_rate(B,h,H,fai,sigma)
I=length(h);
result=zeros(I,1);
for i=1:I
    result(i)=B*log2(1+abs(h(i)+H(:,i)'*fai)^2/sigma^2);
end
end