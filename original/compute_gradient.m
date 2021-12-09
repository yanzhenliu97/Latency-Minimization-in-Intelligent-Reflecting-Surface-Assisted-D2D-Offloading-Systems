function [result]=compute_gradient(B,h,H,fai,sigma,weight,L,k,theta,rate,a)
[M,I]=size(H);
result=zeros(M,1);

for i=1:I
    gradient_out=-weight(i)*L(i)/(k(i)+rate(i))^2;
    gradient_inner=B/(sigma^2+abs(h(i)+H(:,i)'*fai)^2)*((h(i)*H(:,i)+H(:,i)*H(:,i)'*fai).*(-1j*exp(-1j*theta))+conj(h(i)*H(:,i)+H(:,i)*H(:,i)'*fai).*(1j*exp(1j*theta)));
    temp=gradient_out*gradient_inner/a(i);
    result=result+temp;
end