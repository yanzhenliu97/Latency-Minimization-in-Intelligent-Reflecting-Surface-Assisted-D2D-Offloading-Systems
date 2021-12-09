M=40; %反射阵元
I=8; %用户数
J=10; %空闲用户数
sigma=sqrt(10)*0.1;
B=2*1e6; %带宽
noise = 10^(-(174+30)/10)*B;
p_u = 24;
p_u = 10^((p_u-30)/10);
scale_factor = sqrt(sigma^2/noise*p_u);

rng(19970908)
Mmax=1;
Ts=80;
Tf=400;

TTSProposedDelay=zeros(Ts,Tf,Mmax);
STSdelay=zeros(Ts,Tf,Mmax);
TTSmtmDelay=zeros(Ts,Tf,Mmax);
TTSnoIRSDelay=zeros(Ts,Tf,Mmax);
TTSrandomIRSDelay=zeros(Ts,Tf,Mmax);

xAxis=zeros(Mmax,1);
%matchedfj=zeros(I,1);

weight=ones(I,1); %用户权重
L=1e6*(ones(I,1)+4*rand(I,1)); %计算比特数
fi=1e9*ones(I,1); 
fj=1e9*0.5*ones(J,1)+2*1e9*rand(J,1);
C=12*ones(I,1);
d1 = 20; d2 = 10;
[I_position,J_position] = generate_position(I,J,d1,d2);


%统计M变化时的平均时延
for testj=1:Mmax
    %M=2^(testj-1);
    %M=5*testj-5;
    M=30;
%     xAxis(testj)=M;
    
    
    %随机初始化方案
%     theta_pro=2*pi*randn(M,1);
%     fai_pro=exp(1j*theta_pro);
[hi,Hi,Hj] = generate_channel(I_position,J_position,M);

h_sin=randn(1,I)+1j*randn(1,I);
H_sin=randn(M,I)+1j*randn(M,I);
k_sin=zeros(I,1);
a_sin=zeros(I,1);

    %选一帧作为初始化
    focusKMWeight=zeros(I,J);
for i=1:I
    for j=1:J
        %计算单个用户时延
        tempHij=conj(Hi(:,i)).*Hj(:,j);
        fai=(hi(j,i))/abs(hi(j,i))*(tempHij./abs(tempHij));
        focusKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai,sigma);
    end
end

match=KM(focusKMWeight);

%计算提出算法的总时延
for i=1:I
    h_sin(i)=hi(match(i),i);
    H_sin(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_sin(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_sin(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
theta = 2*pi*randn(M,1);
[delay1,optimized_theta]=reflecting(a_sin,k_sin,h_sin,H_sin,sigma,B,theta,weight,L);
theta_pro=optimized_theta;
fai_pro=exp(1j*theta_pro);

    
    theta_mtm=2*pi*randn(M,1);
    fai_mtm = exp(1j*theta_mtm);    
    

    
    fai_noirs=zeros(M,1);
    
    f_theta_pro = zeros(size(theta_pro));
    f_theta_mtm = zeros(size(theta_pro));
    
    for testt=1:Tf
       testt 


    parfor testi=1:Ts
         %initialize the variables of different scheme
%proposed
h_pro=randn(1,I)+1j*randn(1,I);
H_pro=randn(M,I)+1j*randn(M,I);
k_pro=zeros(I,1);
a_pro=zeros(I,1);
%single time scale
h_sin=randn(1,I)+1j*randn(1,I);
H_sin=randn(M,I)+1j*randn(M,I);
k_sin=zeros(I,1);
a_sin=zeros(I,1);
%max to max matching
h_mtm=randn(1,I)+1j*randn(1,I);
H_mtm=randn(M,I)+1j*randn(M,I);
k_mtm=zeros(I,1);
a_mtm=zeros(I,1);
%random_irs_matching
h_ran=randn(1,I)+1j*randn(1,I);
H_ran=randn(M,I)+1j*randn(M,I);
k_ran=zeros(I,1);
a_ran=zeros(I,1);
%no irs matching
h_noi=randn(1,I)+1j*randn(1,I);
H_noi=randn(M,I)+1j*randn(M,I);
k_noi=zeros(I,1);
a_noi=zeros(I,1);

matchedfj=zeros(I,1);
hmatchedfj=zeros(I,1);

[hi,Hi,Hj] = generate_channel(I_position,J_position,M);

hi = hi*scale_factor;
Hi = Hi*sqrt(scale_factor);
Hj = Hj*sqrt(scale_factor);


%scheme 1=＝＝＝＝＝
%proposed scheme:using long term fai to compute the delay between every
%user pair i j and using KM to match them.
randKMWeight=zeros(I,J);
for i=1:I
    for j=1:J
        %计算单个用户时延
        randKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai_pro,sigma);
    end
end

match=KM(randKMWeight);
for i=1:I
    h_pro(i)=hi(match(i),i);
    H_pro(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_pro(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_pro(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
rate=compute_rate(B,h_pro,H_pro,fai_pro,sigma);
delay1=sum(weight.*L./(k_pro+rate)./a_pro);
delay2=sum(weight.*L.*C./(fi+matchedfj));
TTSProposedDelay(testi,testt,testj)=delay1+delay2;


%scheme 2
%single timescale================
%single time scale scheme: The matching matrix and the reflecting
%beamforming matrix is designed in every time slot
focusKMWeight=zeros(I,J);
for i=1:I
    for j=1:J
        %计算单个用户时延
        tempHij=conj(Hi(:,i)).*Hj(:,j);
        fai=(hi(j,i))/abs(hi(j,i))*(tempHij./abs(tempHij));
        focusKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai,sigma);
    end
end

match=KM(focusKMWeight);

%计算提出算法的总时延
for i=1:I
    h_sin(i)=hi(match(i),i);
    H_sin(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_sin(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_sin(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
theta = 2*pi*randn(M,1);
[delay1,rate]=reflecting(a_sin,k_sin,h_sin,H_sin,sigma,B,theta,weight,L);
delay2=weight.*L.*C./(fi+matchedfj);
STSdelay(testi,testt,testj)=delay1+sum(delay2);

%scheme 3
%max to max scheme==================================
%match the max tasks user i to max computation resource user j
heuristicMatch=zeros(I,1);

[sortedL,sortLIndex]=sort(L,'descend');
[sortedFj,sortedFjIndex]=sort(fj,'descend');
for i=1:I
    heuristicMatch(sortLIndex(i))=sortedFjIndex(i);
end

for i=1:I
    h_mtm(i)=hi(heuristicMatch(i),i);
    H_mtm(:,i)=conj(Hi(:,i)).*Hj(:,heuristicMatch(i));
    k_mtm(i)=fi(i)*fj(heuristicMatch(i))/(C(i)*(fi(i)+fj(heuristicMatch(i))));
    a_mtm(i)=(1+fi(i)/fj(heuristicMatch(i)))^2;
    hmatchedfj(i)=fj(heuristicMatch(i));
end

%[delay1,rateh]=reflecting(ah,kh,h,H,sigma,B,theta,weight,L);
rate=compute_rate(B,h_mtm,H_mtm,fai_mtm,sigma);
delay1=sum(weight.*L./(k_mtm+rate)./a_mtm);
delay2=weight.*L.*C./(fi+hmatchedfj);
TTSmtmDelay(testi,testt,testj)=delay1+sum(delay2);


%scheme 4======================================================
%there is no irs
noirsKMWeight=zeros(I,J);

for i=1:I
    for j=1:J
        %计算单个用户时延
        noirsKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai_noirs,sigma);
    end
end

match=KM(noirsKMWeight);
for i=1:I
    h_noi(i)=hi(match(i),i);
    H_noi(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_noi(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_noi(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
rate=compute_rate(B,h_noi,H_noi,fai_noirs,sigma);
delay1=sum(weight.*L./(k_noi+rate)./a_noi);
delay2=sum(weight.*L.*C./(fi+matchedfj));
TTSnoIRSDelay(testi,testt,testj)=delay1+delay2;

%scheme 5=============================
%using the random irs
theta_random_irs=2*pi*randn(M,1);
fai_random_irs = exp(1j*theta_random_irs);
    
randomirsKMWeight=zeros(I,J);
for i=1:I
    for j=1:J
        %计算单个用户时延
        randomirsKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai_random_irs,sigma);
    end
end

match=KM(randomirsKMWeight);
for i=1:I
    h_ran(i)=hi(match(i),i);
    H_ran(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_ran(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_ran(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
rate=compute_rate(B,h_ran,H_ran,fai_random_irs,sigma);
delay1=sum(weight.*L./(k_ran+rate)./a_ran);
delay2=sum(weight.*L.*C./(fi+matchedfj));
TTSrandomIRSDelay(testi,testt,testj)=delay1+delay2;

%time slot end ============================================
  end

%Update the beamforming matrix at the end of the frame=============================
%proposed
rho_t = testt^(-0.60);
gamma_t = testt^(-0.9);
varpi = 5e-4;


%额外产生一组变量用于online的更新
[hi,Hi,Hj] = generate_channel(I_position,J_position,M);

hi = hi*scale_factor;
Hi = Hi*sqrt(scale_factor);
Hj = Hj*sqrt(scale_factor);

randKMWeight=zeros(I,J);
for i=1:I
    for j=1:J
        %计算单个用户时延
        randKMWeight(i,j)=-computeSingleDelay(weight(i),L(i),fi(i),fj(j),C(i),B,hi(j,i),Hi(:,i),Hj(:,j),fai_pro,sigma);
    end
end
h_pro=randn(1,I)+1j*randn(1,I);
H_pro=randn(M,I)+1j*randn(M,I);
k_pro=zeros(I,1);
a_pro=zeros(I,1);

match=KM(randKMWeight);
for i=1:I
    h_pro(i)=hi(match(i),i);
    H_pro(:,i)=conj(Hi(:,i)).*Hj(:,match(i));
    k_pro(i)=fi(i)*fj(match(i))/(C(i)*(fi(i)+fj(match(i))));
    a_pro(i)=(1+fi(i)/fj(match(i)))^2;
    matchedfj(i)=fj(match(i));
end
h_pro_bar=h_pro;
H_pro_bar=H_pro;
k_pro_bar=k_pro;
a_pro_bar=a_pro;

rate=compute_rate(B,h_pro_bar,H_pro_bar,fai_pro,sigma);
g_theta_pro = compute_gradient(B,h_pro_bar,H_pro_bar,fai_pro,sigma,weight,L,k_pro_bar,theta_pro,rate,a_pro_bar);
f_theta_pro = (1-rho_t)*f_theta_pro + rho_t*g_theta_pro;
theta_pro_bar = theta_pro - f_theta_pro/(2*varpi);
theta_pro = (1-gamma_t)*theta_pro + gamma_t*theta_pro_bar;
fai_pro = exp(1j*theta_pro);

%max to max
rho_mtm_t = testt^(-0.6);
gamma_mtm_t = testt^(-0.9);
varpi_mtm = 8e-5;


heuristicMatch=zeros(I,1);
[sortedL,sortLIndex]=sort(L,'descend');
[sortedFj,sortedFjIndex]=sort(fj,'descend');
for i=1:I
    heuristicMatch(sortLIndex(i))=sortedFjIndex(i);
end
h_mtm=randn(1,I)+1j*randn(1,I);
H_mtm=randn(M,I)+1j*randn(M,I);
k_mtm=zeros(I,1);
a_mtm=zeros(I,1);

for i=1:I
    h_mtm(i)=hi(heuristicMatch(i),i);
    H_mtm(:,i)=conj(Hi(:,i)).*Hj(:,heuristicMatch(i));
    k_mtm(i)=fi(i)*fj(heuristicMatch(i))/(C(i)*(fi(i)+fj(heuristicMatch(i))));
    a_mtm(i)=(1+fi(i)/fj(heuristicMatch(i)))^2;
    hmatchedfj(i)=fj(heuristicMatch(i));
end
h_mtm_bar=h_mtm;
H_mtm_bar=H_mtm;
k_mtm_bar=k_mtm;
a_mtm_bar=a_mtm;

rate=compute_rate(B,h_mtm_bar,H_mtm_bar,fai_mtm,sigma);
g_theta_mtm = compute_gradient(B,h_mtm_bar,H_mtm_bar,fai_mtm,sigma,weight,L,k_mtm_bar,theta_mtm,rate,a_mtm_bar);
f_theta_mtm = (1-rho_mtm_t)*f_theta_mtm + rho_mtm_t*g_theta_mtm;
theta_mtm_bar = theta_mtm - f_theta_mtm/(2*varpi_mtm);
theta_mtm = (1-gamma_mtm_t)*theta_mtm + gamma_mtm_t*theta_mtm_bar;
fai_mtm = exp(1j*theta_mtm);



end
% k
% rate
end

figure()
plot(1:Tf,mean(TTSProposedDelay(:,:,1),1))
hold on
plot(1:Tf,mean(STSdelay(:,:,1),1))

% avgDelay_proposed=mean(reshape(TTSProposedDelay(:,Tf,:),Ts,Mmax))*1e3;
% avgDelay_mtm=mean(reshape(TTSmtmDelay(:,Tf,:),Ts,Mmax))*1e3;
% avgDelay_sts=mean(reshape(STSdelay(:,Tf,:),Ts,Mmax))*1e3;
% avgDelay_randomirs=mean(reshape(TTSrandomIRSDelay(:,Tf,:),Ts,Mmax))*1e3;
% avgDelay_noirs=mean(reshape(TTSnoIRSDelay(:,Tf,:),Ts,Mmax))*1e3;
% 
% figure()
% box on
% size = 12
% set(gca,'fontsize',size)
% set(gca,'Xcolor','k')
% xlabel('Reflecting elements M','fontsize',size)
% ylabel('System delay (ms)','fontsize',size)
% % set(gca,'XLim',[0 50]);
% % set(gca,'xTick',[0:10:50])
% % set(gca,'YLim',[180 201]);
% % set(gca,'yTick',[180:3:201]);
% % 
% proposed_plot=line(xAxis,avgDelay_proposed);
% set(proposed_plot,'color','b')
% set(proposed_plot,'linestyle','-')
% set(proposed_plot,'linewidth',1.5)
% set(proposed_plot,'marker','o')
% hold on
% heuristic_plot=line(xAxis,avgDelay_mtm);
% set(heuristic_plot,'color','c')
% set(heuristic_plot,'linestyle','-')
% set(heuristic_plot,'linewidth',1.5)
% set(heuristic_plot,'marker','s')
% hold on
% lowerbound_plot=line(xAxis,avgDelay_sts);
% set(lowerbound_plot,'color','k')
% set(lowerbound_plot,'linestyle','-')
% set(lowerbound_plot,'linewidth',1.5)
% set(lowerbound_plot,'marker','d')
% hold on
% randomirs_plot=line(xAxis,avgDelay_randomirs);
% set(randomirs_plot,'color','m')
% set(randomirs_plot,'linestyle','-')
% set(randomirs_plot,'linewidth',1.5)
% set(randomirs_plot,'marker','>')
% hold on
% noirs_plot=line(xAxis,avgDelay_noirs);
% set(noirs_plot,'color',[1,0.5,0])
% set(noirs_plot,'linestyle','-')
% set(noirs_plot,'linewidth',1.5)
% set(noirs_plot,'marker','<')
% 
% grid on
% legend([proposed_plot,heuristic_plot,lowerbound_plot,randomirs_plot,noirs_plot],'Proposed','Max to max','STS','Random IRS','No IRS','fontsize',9)
% 
