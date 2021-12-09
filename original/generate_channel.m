function [hi,Hi,Hj] = generate_channel(I_position,J_position,M)
%M=64;
%load position

Ny = sqrt(M);
I = size(I_position,2);
J = size(J_position,2);
irs_position = [0,0,3]';

dist_I_irs = zeros(1,I);
dist_J_irs = zeros(1,J);
dist_I_J = zeros(I,J);

AoA_I_irs = zeros(1,I);
AoA_irs_J = zeros(1,J);

for i=1:I
    dist_I_irs(i) = norm(I_position(:,i)-irs_position,'fro');
    eve_AOD_I_IRS = atan((irs_position(3)-I_position(3,i))/norm(irs_position(1:2)-I_position(1:2,i),'fro'));
    azi_AOD_I_IRS = atan((irs_position(1)-I_position(1,i))/(irs_position(2)-I_position(2,i)));
    if (irs_position(1)-I_position(1,i))<0
        azi_AOD_I_IRS=azi_AOD_I_IRS+pi;
    end
    H_I_irs_bar(:,i) =  exp(1j*2*pi/4*(floor((1:M)/Ny)*cos(pi-eve_AOD_I_IRS)+((1:M)-floor((1:M)/Ny)*Ny)*sin(pi-eve_AOD_I_IRS)*cos(pi-azi_AOD_I_IRS)));
    
    for j=1:J
        dist_I_J(i,j) = norm(I_position(:,i)-J_position(:,j),'fro');
    end
end

for j=1:J
    dist_J_irs(j) = norm(J_position(:,j)-irs_position,'fro');
    eve_AOD_IRS_J = atan((irs_position(3)-J_position(3,j))/norm(irs_position(1:2)-J_position(1:2,j),'fro'));
    azi_AOD_IRS_J = atan((irs_position(1)-J_position(1,j))/(irs_position(2)-J_position(2,j)));
    if (irs_position(1)-J_position(1,j))<0
        azi_AOD_J_IRS=azi_AOD_IRS_J+pi;
    end
    H_J_irs_bar(:,j) =  exp(1j*2*pi/4*(floor((1:M)/Ny)*cos(pi-eve_AOD_IRS_J)+((1:M)-floor((1:M)/Ny)*Ny)*sin(pi-eve_AOD_IRS_J)*cos(pi-azi_AOD_IRS_J)));
end

r = 0; % correlation coefficient for IRS
Phi_r = zeros(M,M);
for ii = 1 : M
    for jj = 1 : M
        if ii <=jj
            Phi_r(ii,jj) = r^(jj-ii);
        else
            Phi_r(ii,jj) = conj(Phi_r(jj,ii));
        end
    end
end

beta = 10^(3/10); %rician factor
alpha_loss_user_irs = 2.2; %loss factor
alpha_loss_user_user = 3.2; %loss factor
C0 = 10^(-3); %10^7

for i=1:I
    pathloss_user_irs = 37 + 30*log10(dist_I_irs(i));
    %Hi(:,i) = my_rician_channel(M,1,Phi_r,beta,H_I_irs_bar(:,i))*sqrt(10^(-pathloss_user_irs/10));
    Hi(:,i) = my_rician_channel(M,1,Phi_r,beta,H_I_irs_bar(:,i))*sqrt(sqrt(C0)*dist_I_irs(i)^(-alpha_loss_user_irs));
end

for j=1:J
    pathloss_user_irs = 37 + 30*log10(dist_J_irs(j));
    %Hj(:,j) = my_rician_channel(M,1,Phi_r,beta,H_J_irs_bar(:,j))*sqrt(10^(-pathloss_user_irs/10));
    Hj(:,j) = my_rician_channel(M,1,Phi_r,beta,H_J_irs_bar(:,j))*sqrt(sqrt(C0)*dist_J_irs(j)^(-alpha_loss_user_irs));
end

whether_wall = J_position(1,:)'*I_position(1,:) <0;
%hi = sqrt(2)*(randn(J,I)+1j*randn(J,I)).*sqrt(10.^(-(37 + 30*log10(dist_I_J') + 18.3*whether_wall)/10));
hi = 1/sqrt(2)*(randn(J,I)+1j*randn(J,I)).*sqrt(C0*dist_I_J'.^(-alpha_loss_user_user));