function [result,optimized_theta]=reflecting(a,k,h,H,sigma,B,theta,weight,L)
[M,I]=size(H);
fai=exp(1j*theta);
%���Բ��֣���������
% for testi=1:8
% M=2^testi; %������Ԫ
% I=1; %�û���
% sigma=1e-3;
% B=10e6; %����
% 
% h=randn(I,1)+1j*randn(I,1);
% H=randn(M,I)+1j*randn(M,I);
% load temp.mat
% theta=2*pi*randn(M,1);
% fai=exp(1j*theta);
% weight=ones(I,1);
% L=1e6*[1:I]';
% k=100e6*ones(I,1);
% a=2*ones(I,1);
%���Բ��ֽ���������������


Imax=100;
object=zeros(1,Imax);

%����ͨ������
rate=compute_rate(B,h,H,fai,sigma);

%������ʱ��
delay=weight.*L./(k+rate)./a;

%�����ݶ�
for i=1:Imax
    object(i)=sum(delay);
gradient=compute_gradient(B,h,H,fai,sigma,weight,L,k,theta,rate,a);
theta=armijo(theta,gradient,B,h,H,sigma,weight,L,k,a);

fai=exp(1j*theta);
rate=compute_rate(B,h,H,fai,sigma);
delay=weight.*L./(k+rate)./a;
end
rate;
optimized_theta=theta;
%���û���׼������
%  fai=h/abs(h)*(H./(abs(H)));
%  rate=compute_rate(B,h,H,fai,sigma)
% delay=weight.*L./(k+rate)./a
% object(Imax)
% plot(object);
%=============
%plot(object)
result=object(Imax);
end