function [rician_channel] = my_rician_channel(m,n,Phi1, K_factor, h_bar)
    g_random = 1/sqrt(2)*(randn(m ,n)+sqrt(-1)*(randn(m, n)));   
    g = h_bar*sqrt( K_factor/(K_factor+1)) + sqrt( 1/(K_factor+1))*Phi1^(1/2)* g_random;  
rician_channel=g;
    
 
