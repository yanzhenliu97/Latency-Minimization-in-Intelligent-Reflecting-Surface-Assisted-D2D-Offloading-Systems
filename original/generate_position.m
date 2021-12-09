function [I_position,J_position] = generate_position(I,J,d1,d2)

%tuning part ================
%  I=6;
%  J=8;
% d1=10;
% d2=20;
%tuning part ================

I_position = d2*(rand(3,I)-0.5);
J_position = d2*(rand(3,J)-0.5);

I_position(3,:) = ones(1,I);
J_position(3,:) = ones(1,J);

%完全随机位置
% for i = 1:I
%     while norm(I_position(1:2,i),'fro') <d1
%         I_position(1:2,i) = d2*(randn(2,I)-0.5);
%     end
% end
% 
% for j = 1:J
%     while norm(J_position(1:2,j),'fro') <d1
%         J_position(1:2,J) = d2*(randn(2,J)-0.5);
%     end
% end

% 两个正方形区域
% for i = 1:I
%         I_position(1,i) = d1*(rand(1,1)-0.5)/2.2-d1/4;
%         I_position(2,i) = d2*(rand(1,1)-0.5);
% end
% 
% for j = 1:J
%         J_position(1,j) = d1*(rand(1,1)-0.5)/2.2+d1/4;
%         J_position(2,j) = d2*(rand(1,1)-0.5);
% end

% 两个圆形区域
for i = 1:I
        radius = d1/2*sqrt(rand(1,1));
        theta = 2*pi*rand(1,1);
        I_position(1,i) = radius*cos(theta)-d2/2;
        I_position(2,i) = radius*sin(theta);
end

for j = 1:J
        radius = d1/2*sqrt(rand(1,1));
        theta = 2*pi*rand(1,1);
        J_position(1,j) = radius*cos(theta)+d2/2;
        J_position(2,j) = radius*sin(theta);
end



% figure
% scatter(I_position(1,:),I_position(2,:));
% hold on
% scatter(J_position(1,:),J_position(2,:));


