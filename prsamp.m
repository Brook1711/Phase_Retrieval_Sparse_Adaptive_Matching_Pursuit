clc,clear,close all

fc=3e10;
lambda=3e8/fc;
delta=lambda/2;
k=2*pi/lambda;

L=10;
N=100;
SNR = 10;
RSR = 15;
thrsd=0.1;

x=zeros(N,1);
x(randperm(N,L))=exp(j*rand(L,1)*2*pi);
C=DFTmatrix(N);
y1=C*x;
ref = randn(N,1) + j*randn(N,1);
ref=ref/norm(ref,2)*norm(y1,2)*10^(RSR/20);
y2=AWGN(y1,SNR);
y3=y2+ref;
y=abs(y3);

[x_est,yibuhuoxiabiaoji] = PRSAMP(C,ref,y,thrsd);

NMSE = norm(x_est-x,2)^2/norm(x,2)^2

function [s,yibuhuoxiabiaoji] = PRSAMP(CombingingMatrix,b,z,thrsd)
    [M,N]=size(CombingingMatrix);
    yibuhuoxiabiaoji=[];
    X=[];
    h_temp=zeros(M,1);
    iternum=0;
    initialcancha=norm(z-abs(b),2)^2;
    cancha=initialcancha;
    while cancha/initialcancha>thrsd
        iternum=iternum+1;
        maxcorrelation=0;
        maxxiabiao=0;
        for v = 1:N
            if length(find(yibuhuoxiabiaoji==v)) == 0
                Ctemptemp=[X,CombingingMatrix(:,v)];
                theta=zeros(M,1);
                for iter = 1:30
                    h=pinv(Ctemptemp)*(z.*exp(j*theta)-b);
                    theta=angle(Ctemptemp*h+b);
                end
                tempcorrelation=abs(h(iternum));
                if tempcorrelation > maxcorrelation
                    maxxiabiao=v;
                    maxcorrelation=tempcorrelation;
                end
            end
        end
        yibuhuoxiabiaoji=[yibuhuoxiabiaoji,maxxiabiao];
        X=[X,CombingingMatrix(:,maxxiabiao)];
        theta=zeros(M,1);
        for iter = 1:100
            h=pinv(X)*(z.*exp(j*theta)-b);
            theta=angle(X*h+b);
        end
        h_temp=X*h;
        cancha=norm(z-abs(h_temp+b),2)^2;
    end
    s=zeros(N,1);
    for u = 1:iternum
        s(yibuhuoxiabiaoji(u))=h(u);
    end
end

function W = DFTmatrix(N)
    W =zeros(N);
    for u = 1:N
        for v = 1:N
            W(u,v)=exp(j*u*v*2*pi/N)/sqrt(N);
        end
    end
end

function x_noise = AWGN(x,SNR)
    [a,b]=size(x);
    nengliang = norm(x,'fro')^2/a/b/10^(SNR/10);
    x_noise = x + randn(size(x))*sqrt(nengliang/2) + j*randn(size(x))*sqrt(nengliang/2);
end