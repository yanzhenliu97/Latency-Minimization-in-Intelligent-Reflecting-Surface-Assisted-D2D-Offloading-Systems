function [result,optimized_theta]=reflecting(a,k,h,H,sigma,B,theta,weight,L)
[M,I]=size(H);
fai=exp(1j*theta);
%调试部分＝＝＝＝＝
% for testi=1:8
% M=2^testi; %反射阵元
% I=1; %用户数
% sigma=1e-3;
% B=10e6; %带宽
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
%调试部分结束＝＝＝＝＝＝


Imax=100;
object=zeros(1,Imax);

%计算通信速率
rate=compute_rate(B,h,H,fai,sigma);

%计算总时延
delay=weight.*L./(k+rate)./a;

%计算梯度
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
%单用户对准＝＝＝
%  fai=h/abs(h)*(H./(abs(H)));
%  rate=compute_rate(B,h,H,fai,sigma)
% delay=weight.*L./(k+rate)./a
% object(Imax)
% plot(object);
%=============
%plot(object)
result=object(Imax);
end