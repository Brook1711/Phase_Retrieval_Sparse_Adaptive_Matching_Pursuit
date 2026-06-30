clc,clear,close all

fc=1.5e10;
Lx=1;
Ly=1;
lambda=3e8/fc;
k=2*pi/lambda;
delta=lambda/2;
Nx=2*round(Lx/delta/2)+1;
Ny=Nx;
ucenter=(Nx+1)/2;
vcenter=(Ny+1)/2;

M=200;
kB=1.38e-23;
bandwidth=1e8;
temperature=290;
sigma=sqrt(kB*bandwidth*temperature/2);

rou=5;
theta=45*pi/180;
phi=30*pi/180;
xx=rou*sin(theta)*cos(phi);
yy=rou*sin(theta)*sin(phi);
zz=rou*cos(theta);

Ha = WDEF(fc,Lx,Ly,rou,theta,phi);
[Ha_vec,zuobiao,edge_start,edge_end] = Ha_1dimensionalize(Ha);

[mmmm,nnnn]=size(Ha);
mmmmm=-1:2/(mmmm-1):1;
nnnnn=-1:2/(nnnn-1):1;
[MMMMM,NNNNN]=meshgrid(mmmmm,nnnnn);

L = length(Ha_vec);

C = randn(M,L) + j*randn(M,L);

y = C*Ha_vec;

y = y + sigma*randn(size(y)) + j*sigma*randn(size(y));


global Ha_temp
Ha_temp = lsqminnorm(C,y);


figure

Ha_2d = NaN*ones(size(Ha));
zuobiao_2d = zeros(size(Ha));

for ll = 1:L
    Ha_2d(zuobiao(ll,1),zuobiao(ll,2)) = Ha_temp(ll);
    zuobiao_2d(zuobiao(ll,1),zuobiao(ll,2)) = ll;
end

surf(MMMMM,NNNNN,abs(Ha_2d));
axis([-1 1 -1 1 0 max(max(abs(Ha)))])
xlabel("$\kappa_x=k_x/k$",'Interpreter','latex','FontSize',14)
ylabel("$\kappa_y=k_y/y$",'Interpreter','latex','FontSize',14)
xticks(-1:0.5:1);
yticks(-1:0.5:1);
view([0 90])
shading flat; 
title("LS Estimation Result",'interpreter','latex');

xiabiaojihe = graphcut_fudu(Ha_temp,edge_start,edge_end);

Ha_cut = zeros(size(Ha_temp));
Ha_cut(xiabiaojihe) = Ha_temp(xiabiaojihe);

xiabiaochangdu = length(xiabiaojihe);

figure

Ha_2d_cut = zeros*ones(size(Ha));
Ha_2d_cut(isnan(Ha_2d)) = NaN;
for ll = 1:xiabiaochangdu
    Ha_2d_cut(zuobiao(xiabiaojihe(ll),1),zuobiao(xiabiaojihe(ll),2)) = Ha_cut(xiabiaojihe(ll));
end

surf(MMMMM,NNNNN,abs(Ha_2d_cut));
axis([-1 1 -1 1 0 max(max(abs(Ha)))])
xlabel("$\kappa_x=k_x/k$",'Interpreter','latex','FontSize',14)
ylabel("$\kappa_y=k_y/y$",'Interpreter','latex','FontSize',14)
xticks(-1:0.5:1);
yticks(-1:0.5:1);
view([0 90])
shading flat; 
title("After Amplitude-Based Graph Cut",'interpreter','latex');

xiabiaojihe_2d = zeros(xiabiaochangdu,2);
for ll = 1:xiabiaochangdu
    xiabiaojihe_2d(ll,1) = zuobiao(xiabiaojihe(ll),1);
    xiabiaojihe_2d(ll,2) = zuobiao(xiabiaojihe(ll),2);
end

xiabiaojihe_2d_guiyihua = xiabiaojihe_2d;
xiabiaojihe_2d_guiyihua(:,1) = (xiabiaojihe_2d_guiyihua(:,1)-ucenter)/((Nx-1)/2);
xiabiaojihe_2d_guiyihua(:,2) = (xiabiaojihe_2d_guiyihua(:,2)-vcenter)/((Ny-1)/2);

scatterplot(xiabiaojihe_2d_guiyihua(:,2)+j*xiabiaojihe_2d_guiyihua(:,1))
title("Extract Discrete Indices",'interpreter','latex')
xlabel("$\kappa_x$",'interpreter','latex');
ylabel("$\kappa_y$",'interpreter','latex');
hold on;
zhuanquan = linspace(0, 2*pi, 1000);
x_circle = cos(zhuanquan);
y_circle = sin(zhuanquan);

plot(x_circle, y_circle, 'r--', 'LineWidth', 1.5);

axis equal;
axis([-1 1 -1 1])

[num_clusters, cluster_indices, noise_indices] = dbscan_clustering(xiabiaojihe_2d, 1.5, 3);

xiabiaojihe_2d_yuanshi = xiabiaojihe_2d;

figure

for nn = 1:num_clusters
    xiabiaojihe_2d = xiabiaojihe_2d_yuanshi(cell2mat(cluster_indices(nn)),:);
    xiabiaojihe_2d_guiyihua = xiabiaojihe_2d;
    xiabiaojihe_2d_guiyihua(:,1) = (xiabiaojihe_2d_guiyihua(:,1)-ucenter)/((Nx-1)/2);
    xiabiaojihe_2d_guiyihua(:,2) = (xiabiaojihe_2d_guiyihua(:,2)-vcenter)/((Ny-1)/2);
    scatter(xiabiaojihe_2d_guiyihua(:,2),xiabiaojihe_2d_guiyihua(:,1));
    hold on;
end

kappacenterx = mean(xiabiaojihe_2d_guiyihua(:,2));
kappacentery = mean(xiabiaojihe_2d_guiyihua(:,1));
xlabel("")
ylabel("")
title("DBSCAN Result without Cluster Selection",'interpreter','latex')
hold on;
xlabel("$\kappa_x$",'interpreter','latex');
ylabel("$\kappa_y$",'interpreter','latex');
zhuanquan = linspace(0, 2*pi, 1000);
x_circle = cos(zhuanquan);
y_circle = sin(zhuanquan);

plot(x_circle, y_circle, 'r--', 'LineWidth', 1.5);

axis equal;
axis([-1 1 -1 1])





xiabiaojihe_2d = xiabiaojihe_2d_yuanshi;

zuidanengliang=0;
nengliangzuidacluster=0;

for nn = 1:num_clusters
    linshi = Ha_temp(xiabiaojihe(cell2mat(cluster_indices(nn))));
    linshipower = norm(linshi,2)^2;
    if linshipower > zuidanengliang
        zuidanengliang = linshipower;
        nengliangzuidacluster = nn;
    end
end

xiabiaojihe_2d = xiabiaojihe_2d(cell2mat(cluster_indices(nengliangzuidacluster)),:);
xiabiaojihe = xiabiaojihe(cell2mat(cluster_indices(nengliangzuidacluster)));

xiabiaojihe_2d_guiyihua = xiabiaojihe_2d;
xiabiaojihe_2d_guiyihua(:,1) = (xiabiaojihe_2d_guiyihua(:,1)-ucenter)/((Nx-1)/2);
xiabiaojihe_2d_guiyihua(:,2) = (xiabiaojihe_2d_guiyihua(:,2)-vcenter)/((Ny-1)/2);

scatterplot(xiabiaojihe_2d_guiyihua(:,2)+j*xiabiaojihe_2d_guiyihua(:,1))
kappacenterx = mean(xiabiaojihe_2d_guiyihua(:,2));
kappacentery = mean(xiabiaojihe_2d_guiyihua(:,1));
xlabel("")
ylabel("")
title("$\mathcal{I}_{\rm sub}$ After DBSCAN Clustering",'interpreter','latex')
hold on;
xlabel("$\kappa_x$",'interpreter','latex');
ylabel("$\kappa_y$",'interpreter','latex');
zhuanquan = linspace(0, 2*pi, 1000);
x_circle = cos(zhuanquan);
y_circle = sin(zhuanquan);

plot(x_circle, y_circle, 'r--', 'LineWidth', 1.5);

axis equal;
axis([-1 1 -1 1])

scatter(kappacenterx, kappacentery, 300, 'p', 'filled', ...
    'MarkerFaceColor', [0.5 0 0.5], ...
    'MarkerEdgeColor', 'none'); 


xxxxxx = xiabiaojihe_2d(:, 1);
yyyyyy = xiabiaojihe_2d(:, 2);

minX = min(xxxxxx); maxX = max(xxxxxx);
minY = min(yyyyyy); maxY = max(yyyyyy);
img = false(maxY - minY + 1, maxX - minX + 1);
idx = sub2ind(size(img), yyyyyy - minY + 1, xxxxxx - minX + 1);
img(idx) = true;

boundaryImg = bwperim(img, 8);

B = bwboundaries(boundaryImg, 8, 'noholes');

boundaryRaw = B{1};  
tubao = [boundaryRaw(:, 2) + minX - 1, boundaryRaw(:, 1) + minY - 1];

fprintf("找到边界\n");
tubao_guiyihua = tubao;
tubao_guiyihua(:,1) = (tubao_guiyihua(:,1)-ucenter)/((Nx-1)/2);
tubao_guiyihua(:,2) = (tubao_guiyihua(:,2)-vcenter)/((Ny-1)/2);

scatterplot(tubao_guiyihua(:,2)+j*tubao_guiyihua(:,1));
xlabel("")
ylabel("")
title("Boundary of $\mathcal{I}_{\rm sub}$ in $\kappa_x$-$\kappa_y$ Domain",'interpreter','latex')
hold on;
xlabel("$\kappa_x$",'interpreter','latex');
ylabel("$\kappa_y$",'interpreter','latex');
zhuanquan = linspace(0, 2*pi, 1000);
x_circle = cos(zhuanquan);
y_circle = sin(zhuanquan);

plot(x_circle, y_circle, 'r--', 'LineWidth', 1.5);

axis equal;
axis([-1 1 -1 1])

boundarynormalized = tubao_guiyihua;

[bianjiedianshu,~]=size(boundarynormalized);
bianhuanhoudianji=zeros(size(bianjiedianshu));
for uu = 1:bianjiedianshu
    bianhuanhoudianji(uu,1)=sqrt((1-boundarynormalized(uu,2)^2)/boundarynormalized(uu,1)^2);
    bianhuanhoudianji(uu,2)=sqrt((1-boundarynormalized(uu,1)^2)/boundarynormalized(uu,2)^2);
end

cc = EM(bianhuanhoudianji);

alpha=1-(cc(2)^2-1)/(cc(4)^2-1);
beta=1+(cc(2)^2-1)/(cc(4)^2-1);
if sign((-sqrt(beta^2-alpha^2)-beta)/alpha/2) == sign((-sqrt(beta^2-alpha^2)-beta)/alpha/2-0.5)
    xu=(-sqrt(beta^2-alpha^2)-beta)/alpha/2*Lx;
else
    xu=(+sqrt(beta^2-alpha^2)-beta)/alpha/2*Lx;
end

chi=1-(cc(1)^2-1)/(cc(3)^2-1);
xi=1+(cc(1)^2-1)/(cc(3)^2-1);
if sign((-sqrt(xi^2-chi^2)-xi)/chi/2) == sign((-sqrt(xi^2-chi^2)-xi)/chi/2-0.5)
    yu=(-sqrt(xi^2-chi^2)-xi)/chi/2*Lx;
else
    yu=(+sqrt(xi^2-chi^2)-xi)/chi/2*Lx;
end

zu=(sqrt(cc(2)^2-1)*(xu-0.5*Lx)+sqrt(cc(1)^2-1)*(yu-0.5*Ly))/2;

[xU, yU, zU] = estimatePosition(cc(2), cc(1), cc(4), cc(3), Lx, Ly);

rounaive = sqrt(xu^2+yu^2+zu^2);
roulsefl = sqrt(xU^2+yU^2+zU^2);
params = [-rounaive/10, -rounaive/5, -rounaive*0.3, -rounaive*0.4, -rounaive/2, -rounaive*0.6];
n = length(params);

xxxx = zeros(1, n);
yyyy = zeros(1, n);
zzzz = zeros(1, n);
canchajihe = zeros(1, 2*n);

for i = 1:n
    [xxxx(i), yyyy(i), zzzz(i), canchajihe(i)] = jingxi(xu, yu, zu, y, C, fc, Lx, Ly, params(i));
end

params = [-roulsefl/10, -roulsefl/5, -roulsefl*0.3, -roulsefl*0.4, -roulsefl/2, -roulsefl*0.6];


for i = 1:n
    [xxxx(i+n), yyyy(i+n), zzzz(i+n), canchajihe(i+n)] = jingxi(xU, yU, zU, y, C, fc, Lx, Ly, params(i));
end

[~, idx] = min(canchajihe);

x1 = xxxx(idx);
y1 = yyyy(idx);
z1 = zzzz(idx);


figure
plot3(bianhuanhoudianji(:,2), bianhuanhoudianji(:,1),2*ones(size(bianhuanhoudianji(:,2))), 'm-', 'LineWidth', 2);
xlabel("$u$",'Interpreter','latex','FontSize',14)
ylabel("$v$",'Interpreter','latex','FontSize',14)
title("$\mathcal{B}$: Map $\kappa_x$-$\kappa_y$ Boundary to $u$-$v$ Domain via $f$",'Interpreter','latex','FontSize',14)
view([0 90])
grid on

figure
plot3(bianhuanhoudianji(:,2), bianhuanhoudianji(:,1),2*ones(size(bianhuanhoudianji(:,2))), 'm-', 'LineWidth', 2);
xlabel("$u$",'Interpreter','latex','FontSize',14)
ylabel("$v$",'Interpreter','latex','FontSize',14)
title("Soft GMM-Based Parametric Estimation",'Interpreter','latex','FontSize',14)
view([0 90])
grid on;
hold on;
ttt=1:1e-2:2.5;
xxx1=cc(2)*ones(size(ttt));
xxx2=cc(4)*ones(size(ttt));
yyy1=cc(1)*ones(size(ttt));
yyy2=cc(3)*ones(size(ttt));
plot3(xxx1, ttt,2*ones(size(ttt)), 'r--', 'LineWidth', 0.5);
plot3(ttt, yyy1,2*ones(size(ttt)), 'color', [0, 180/255, 75/255], 'LineWidth', 0.5, 'LineStyle','--');
plot3(xxx2, ttt,2*ones(size(ttt)), 'color', [1, 192/255, 0], 'LineWidth', 0.5, 'LineStyle','--');
plot3(ttt, yyy2,2*ones(size(ttt)), 'color', [51/255, 82/255,170/255], 'LineWidth', 0.5, 'LineStyle','--');
xticks(sort([min([cc(4),cc(2)]-0.5) cc(4) cc(2) max([cc(4),cc(2)]+0.5)],'ascend'))
yticks(sort([min([cc(3),cc(1)]-0.5) cc(3) cc(1) max([cc(3),cc(1)]+0.5)],'a'))
axis([min([cc(4),cc(2)]-0.5) max([cc(4),cc(2)]+0.5) min([cc(3),cc(1)]-0.5) max([cc(3),cc(1)]+0.5)])

figure
scatter3(xx, yy, zz, 200, 'ro', 'filled', 'MarkerEdgeColor', 'r', 'markerfacecolor','r','LineWidth', 2, 'MarkerFaceAlpha', 0.1);
hold on;
plot3(xu,yu,zu,'b*', 'MarkerSize', 10, 'LineWidth', 2);
plot3(xU,yU,zU,'k^', 'MarkerSize', 10, 'LineWidth', 2);
plot3(x1,y1,z1,'ms', 'MarkerSize', 10, 'LineWidth', 2);
grid on;
axis([-1 10 -1 10 0 10])
xlabel("$x$",'Interpreter','latex','FontSize',14)
ylabel("$y$",'Interpreter','latex','FontSize',14)
zlabel("$z$",'Interpreter','latex','FontSize',14)
x_mimo = [-0.5, 0.5, 0.5, -0.5];
y_mimo = [-0.5, -0.5, 0.5, 0.5];
z_mimo = [0, 0, 0, 0]; 
fill3(x_mimo, y_mimo, z_mimo, [0.7 0.7 0.7], 'EdgeColor', 'none');

legend("Precise Location","Naive ELF Rough Estimate","LS-ELF Rough Estimate","After Gradient-Descent","UPA Array",'Interpreter','latex','FontSize',14)
title("Positioning Result via CV-EFL",'interpreter','latex')



function Ha = WDEF(fc,Lx,Ly,rou,theta,phi)

lambda=3e8/fc;
k=2*pi/lambda;
delta=lambda/2;
Nx=2*round(Lx/delta/2)+1;
Ny=Nx;
ucenter=(Nx+1)/2;
vcenter=(Ny+1)/2;

xx=rou*sin(theta)*cos(phi);
yy=rou*sin(theta)*sin(phi);
zz=rou*cos(theta);

H=zeros(Nx,Ny);

u = (1:Nx)';
v = 1:Ny;

u_offset = (u - ucenter) * delta;  
v_offset = (v - vcenter) * delta;  

juli = sqrt( (xx - u_offset).^2 + (yy - v_offset).^2 + zz^2 );

H = exp(-1j * k * juli) ./ juli;

Ha=(fftshift(fft2(H)))';

for u = 1:Nx
    for v = 1:Ny
        if (u-ucenter)^2+(v-vcenter)^2>ucenter^2
            Ha(u,v)=NaN;
        end
    end
end

[mm,nn]=size(Ha);

Ha_complex = Ha;
Ha = abs(Ha);

mmm=-1:2/(mm-1):1;
nnn=-1:2/(nn-1):1;
[M,N]=meshgrid(mmm,nnn);
surf(M,N,abs(Ha));
axis([-1 1 -1 1 0 max(max(abs(Ha)))])
xlabel("$\kappa_x=k_x/k$",'Interpreter','latex','FontSize',14)
ylabel("$\kappa_y=k_y/y$",'Interpreter','latex','FontSize',14)
xticks(-1:0.5:1);
yticks(-1:0.5:1);
title("True Wavenumber-Domain Spectrum",'Interpreter','latex','FontSize',14)
view([0 90])
hold on;
shading flat; 

a1 = sqrt((xx-0.5*Lx)^2/((xx-0.5*Lx)^2+zz^2)); 
b1 = 1; 
drawEllipse(a1, b1, 'r', 2, max(max(Ha)))

a2 = sqrt((xx+0.5*Lx)^2/((xx+0.5*Lx)^2+zz^2)); 
b2 = 1; 
drawEllipse(a2, b2, [0, 180/255, 75/255], 2, max(max(Ha)))

a3 = 1; 
b3 = sqrt((yy-0.5*Ly)^2/((yy-0.5*Ly)^2+zz^2)); 
drawEllipse(a3, b3, [1, 192/255,0], 2, max(max(Ha)))

a4 = 1; 
b4 = sqrt((yy+0.5*Ly)^2/((yy+0.5*Ly)^2+zz^2)); 
drawEllipse(a4, b4, [51/255, 82/255,0/255], 2, max(max(Ha)))

Ha =Ha_complex;
end

function drawEllipse(a, b, color, lineWidth, maxzhi)
    center=[0,0];
    theta=0;
    t = linspace(0, 2*pi, 361);
    x = center(1) + a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
    y = center(2) + a * cos(t) * sin(theta) + b * sin(t) * cos(theta);
    plot3(x, y, maxzhi*ones(size(x))/5, 'Color', color, 'LineWidth', 0.5, 'LineStyle', '--');
end

function x_noise = AWGNreal(x,SNR)
    [a,b]=size(x);
    zongnengliang=sum(sum(x.^2));
    danweinengliang=zongnengliang/a/b;
    zaoshengnengliang=danweinengliang/10^(SNR/10);
    x_noise=x+randn(a,b)*sqrt(zaoshengnengliang);
end

function [boundary, largestRegion] = findLargestNonzeroBoundary(matrix,Ha)

    binaryMatrix = matrix ~= 0;
    
    cc = bwconncomp(binaryMatrix);
    
    if cc.NumObjects == 0
        boundary = [];
        largestRegion = [];
        return;
    end
    
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = max(numPixels);
    largestRegion = zeros(size(binaryMatrix));
    largestRegion(cc.PixelIdxList{idx}) = 1;
    
    boundary = bwboundaries(largestRegion);
    boundary = boundary{1}; 

    [mm,nn]=size(matrix);
    mmm=-1:2/(mm-1):1;
    nnn=-1:2/(nn-1):1;
    [M,N]=meshgrid(mmm,nnn);
    
    figure;
    matrix(find(isnan(Ha)==1))=NaN;
    mesh(M,N,matrix);
    axis([-1 1 -1 1 0 max(max(Ha))])
xlabel("$\kappa_x=k_x/k$",'Interpreter','latex','FontSize',14)
ylabel("$\kappa_y=k_y/k$",'Interpreter','latex','FontSize',14)
xticks(-1:0.5:1);
yticks(-1:0.5:1);
title("$\mathbf{h}_a^{\rm (Filtered)}$ under SNR=0 dB",'Interpreter','latex','FontSize',14)
view([0 90])
hold on;
    
end

function [num_clusters, cluster_indices, noise_indices] = dbscan_clustering(data, eps, minPts)

    labels = dbscan(data, eps, minPts);
    
    unique_labels = unique(labels);
    unique_labels = unique_labels(unique_labels ~= -1);  
    num_clusters = length(unique_labels);
    
    cluster_indices = cell(1, num_clusters);
    for i = 1:num_clusters
        cluster_indices{i} = find(labels == unique_labels(i));
    end
    
    noise_indices = find(labels == -1);
end

function c = EM(bianhuanhoudianji)
[dianshu,~]=size(bianhuanhoudianji);
sigma2=0.01*[1,1,1,1];
c=zeros(1,4);
gaoxiao = sort(bianhuanhoudianji(:,1),'ascend');
c(1)=mean(gaoxiao(round(dianshu/2):end));
gaoxiao = sort(bianhuanhoudianji(:,2),'ascend');
c(2)=mean(gaoxiao(round(dianshu/2):end));
gaoxiao = sort(bianhuanhoudianji(:,1),'descend');
c(3)=mean(gaoxiao(round(dianshu/2):end));
gaoxiao = sort(bianhuanhoudianji(:,2),'descend');
c(4)=mean(gaoxiao(round(dianshu/2):end));
gailvmidu = zeros(4,dianshu);
gailvmidushangyici = ones(4,dianshu);
for itertimes = 1:100
    gailvmidushangyici=gailvmidu;
    for u = 1:dianshu
        gailvmidu(1,u)=1/sqrt(sigma2(1))*exp(-(bianhuanhoudianji(u,1)-c(1))^2/2/sigma2(1));
        gailvmidu(2,u)=1/sqrt(sigma2(2))*exp(-(bianhuanhoudianji(u,2)-c(2))^2/2/sigma2(2));
        gailvmidu(3,u)=1/sqrt(sigma2(3))*exp(-(bianhuanhoudianji(u,1)-c(3))^2/2/sigma2(3));
        gailvmidu(4,u)=1/sqrt(sigma2(4))*exp(-(bianhuanhoudianji(u,2)-c(4))^2/2/sigma2(4));
        gailvmidu(:,u)=gailvmidu(:,u)/sum(gailvmidu(:,u));
    end
    for u = 1:4
        if u == 1 || u == 3
            quanzhong = gailvmidu(u,:)';
            quanzhong = quanzhong/sum(quanzhong);
            shuzu = bianhuanhoudianji(:,1);
            c(u) = sum(shuzu .* quanzhong);
            sigma2(u)=sum(quanzhong .* (shuzu - c(u)).^2);
        else
            quanzhong = gailvmidu(u,:)';
            quanzhong = quanzhong/sum(quanzhong);
            shuzu = bianhuanhoudianji(:,2);
            c(u) = sum(shuzu .* quanzhong);
            sigma2(u)=sum(quanzhong .* (shuzu - c(u)).^2);
        end
    end
    if norm(gailvmidu-gailvmidushangyici,'fro')^2/norm(gailvmidushangyici,'fro')^2 < 1e-6
        break;
    end
end
end



function [Ha_vec, zuobiao, edge_start, edge_end] = Ha_1dimensionalize(Ha)
    [mm, nn] = size(Ha);
    valid_mask = ~isnan(Ha);
    [zuobiao(:,1), zuobiao(:,2)] = find(valid_mask);
    Ha_vec = Ha(valid_mask);
    
    num_valid = length(Ha_vec);
    idxMat = zeros(mm, nn);
    idxMat(valid_mask) = 1:num_valid;
    
    [r, c] = find(valid_mask & [valid_mask(:,2:end), false(mm,1)]);
    edge_right = [idxMat(sub2ind([mm,nn], r, c)), idxMat(sub2ind([mm,nn], r, c+1))];
    
    [r, c] = find(valid_mask & [valid_mask(2:end,:); false(1,nn)]);
    edge_down = [idxMat(sub2ind([mm,nn], r, c)), idxMat(sub2ind([mm,nn], r+1, c))];
    
    edges = [edge_right; edge_down];
    edge_start = [edges(:,1); edges(:,2)];
    edge_end   = [edges(:,2); edges(:,1)];
end


function xiabiaojihe = graphcut_fudu(Ha_temp,edge_start,edge_end)
    n = length(Ha_temp);    
    Ha_temp = Ha_temp / norm(Ha_temp,2) * sqrt(n);
    cap_source = abs(Ha_temp);
    cap_sink = 1./cap_source;
    edge_start = edge_start';
    edge_end = edge_end';
    concentrationrate=10; 
    for iter = 1:30
        cap_xianglin=concentrationrate;
        [set_index_source,set_index_sink]=zuixiaoge(cap_source,cap_sink,cap_xianglin,edge_start,edge_end);
        if length(set_index_source)==n || length(set_index_source)<=1
            concentrationrate=concentrationrate/2;
        else
            break;
        end
    end
    xiabiaojihe = set_index_source;
end


function [set_index_source, set_index_sink] = zuixiaoge(cap_source, cap_sink, cap_xianglin, edge_tou, edge_wei)
    n_node = length(cap_sink);
    
    s = n_node + 1;  
    t = n_node + 2; 
    
    all_s = [];
    all_t = [];
    all_weights = [];
    
    all_s = [all_s, repmat(s, 1, n_node)];
    all_t = [all_t, 1:n_node];
    all_weights = [all_weights, cap_source'];
    
    all_s = [all_s, 1:n_node];
    all_t = [all_t, repmat(t, 1, n_node)];
    all_weights = [all_weights, cap_sink'];
    
    all_s = [all_s, edge_tou];
    all_t = [all_t, edge_wei];

    all_weights = [all_weights, repmat(cap_xianglin, 1, length(edge_tou))];
    
    G = digraph(all_s, all_t, all_weights);
    
    [~, ~, cs, ct] = maxflow(G, s, t);
    
    set_index_source = cs(cs <= n_node); 
    set_index_sink = ct(ct <= n_node);   
    
    set_index_source = sort(set_index_source);
    set_index_sink = sort(set_index_sink);
end









function cancha = cost(xx,yy,zz,y,C,fc,Lx,Ly)
global Ha_temp
    lambda=3e8/fc;
k=2*pi/lambda;
delta=lambda/2;
Nx=2*round(Lx/delta/2)+1;
Ny=Nx;
ucenter=(Nx+1)/2;
vcenter=(Ny+1)/2;


H=zeros(Nx,Ny);

u = (1:Nx)';
v = 1:Ny;

u_offset = (u - ucenter) * delta;  
v_offset = (v - vcenter) * delta;  

juli = sqrt( (xx - u_offset).^2 + (yy - v_offset).^2 + zz^2 );

H = exp(-1j * k * juli) ./ juli;

Ha=(fftshift(fft2(H)))';

for u = 1:Nx
    for v = 1:Ny
        if (u-ucenter)^2+(v-vcenter)^2>ucenter^2
            Ha(u,v)=NaN;
        end
    end
end
[mm, nn] = size(Ha);
    valid_mask = ~isnan(Ha);
    [zuobiao(:,1), zuobiao(:,2)] = find(valid_mask);
    Ha_vec = Ha(valid_mask);


    Ha_vec = C' * ((C * C') \ (C * Ha_vec));


    cancha = norm(abs(Ha_vec)-abs(Ha_temp),2)^2;
end

function [x1,y1,z1,cancha] = jingxi(xx,yy,zz,y,C,fc,Lx,Ly,roupianli)
    itertimes = 10;
    jiaoshitan = pi/180;
    rou = sqrt(xx^2+yy^2+zz^2);
    theta = acos(zz/rou);
    phi = atan(yy/xx);
    rou = rou+roupianli;
    roushitan = min(1e-3,1/rou/100);
    for iter = 1:itertimes
        daijia = cost(xx,yy,zz,y,C,fc,Lx,Ly);
        weizhi = [xx,yy,zz];
        fprintf("Current cost is %f\n",daijia);
        routemp = 1/(1/rou - roushitan);
        dfdrou = cost(xx,yy,zz,y,C,fc,Lx,Ly)-cost(routemp*cos(phi)*sin(theta),routemp*sin(phi)*sin(theta),routemp*cos(theta),y,C,fc,Lx,Ly);
        dfdrou = dfdrou / norm(dfdrou,2);
        buchang = min(0.1,1/rou/10);
        for amj = 1:10
            routemp = 1/(1/rou + dfdrou*buchang);
            xxtemp = routemp*cos(phi)*sin(theta);
            yytemp = routemp*sin(phi)*sin(theta);
            zztemp = routemp*cos(theta);
            daijiatemp = cost(xxtemp,yytemp,zztemp,y,C,fc,Lx,Ly);
            if daijiatemp < daijia
                xx = xxtemp;
                yy = yytemp;
                zz = zztemp;
                rou = routemp;
                daijia = daijiatemp;
                break;
            end
            buchang = buchang/2;
        end
        if amj >= 10
            fprintf("break\n");
            break;
        end
        thetatemp = theta - jiaoshitan;
        dfdtheta = cost(xx,yy,zz,y,C,fc,Lx,Ly)-cost(rou*cos(phi)*sin(thetatemp),rou*sin(phi)*sin(thetatemp),rou*cos(thetatemp),y,C,fc,Lx,Ly);
        dfdtheta = dfdtheta / norm(dfdtheta,2);
        buchang = pi/90;
        for amj = 1:10
            thetatemp = theta - dfdtheta*buchang;
            xxtemp = rou*cos(phi)*sin(thetatemp);
            yytemp = rou*sin(phi)*sin(thetatemp);
            zztemp = rou*cos(thetatemp);
            daijiatemp = cost(xxtemp,yytemp,zztemp,y,C,fc,Lx,Ly);
            if daijiatemp < daijia
                xx = xxtemp;
                yy = yytemp;
                zz = zztemp;
                theta = thetatemp;
                daijia = daijiatemp;
                break;
            end
            buchang = buchang/2;
        end
        if amj >= 10
            fprintf("break\n");
            break;
        end
        phitemp = phi - jiaoshitan;
        dfdphi = cost(xx,yy,zz,y,C,fc,Lx,Ly)-cost(rou*cos(phitemp)*sin(theta),rou*sin(phitemp)*sin(theta),rou*cos(theta),y,C,fc,Lx,Ly);
        dfdphi = dfdphi / norm(dfdphi,2);
        buchang = pi/90;
        for amj = 1:10
            phitemp = phi - dfdphi*buchang;
            xxtemp = rou*cos(phitemp)*sin(theta);
            yytemp = rou*sin(phitemp)*sin(theta);
            zztemp = rou*cos(theta);
            daijiatemp = cost(xxtemp,yytemp,zztemp,y,C,fc,Lx,Ly);
            if daijiatemp < daijia
                xx = xxtemp;
                yy = yytemp;
                zz = zztemp;
                phi = phitemp;
                daijia = daijiatemp;
                break;
            end
            buchang = buchang/2;
        end
        if amj >= 10
            fprintf("break\n");
            break;
        end
    end
    x1 = rou*cos(phi)*sin(theta);
    y1 = rou*sin(phi)*sin(theta);
    z1 = rou*cos(theta);
    cancha = cost(x1,y1,z1,y,C,fc,Lx,Ly);

end

function [xU, yU, zU] = estimatePosition(mu1, mu2, mu3, mu4, Lx, Ly)

    mu1 = mu1(:); mu2 = mu2(:); mu3 = mu3(:); mu4 = mu4(:);
    N = length(mu1);
    
    s1 = sqrt(mu1.^2 - 1);
    s2 = sqrt(mu2.^2 - 1);
    s3 = sqrt(mu3.^2 - 1);
    s4 = sqrt(mu4.^2 - 1);
    
    xU = zeros(N, 1);
    yU = zeros(N, 1);
    zU = zeros(N, 1);
    
    for i = 1:N
        A = [s1(i),    0,   -1;
             0,    s2(i),   -1;
             s3(i),   0,    -1;
             0,    s4(i),   -1];
        
        b = [0.5*Lx*s1(i);
             0.5*Ly*s2(i);
             -0.5*Lx*s3(i);
             -0.5*Ly*s4(i)];
        
        p = (A'*A) \ (A'*b);
        
        xU(i) = p(1);
        yU(i) = p(2);
        zU(i) = p(3);
    end
end