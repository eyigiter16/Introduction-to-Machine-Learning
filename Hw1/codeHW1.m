clc
clear all;
close all;

%parameters:
    %class means:
    classMean1 = [0.0 2.5];
    classMean2 = [-2.5 -2.0];
    classMean3 = [2.5 -2.0];
    %class deviations:
    classDev1 = [3.2 0.0; 0.0 1.2];
    classDev2 = [1.2 -0.8; -0.8 1.2];
    classDev3 = [1.2 0.8; 0.8 1.2];
    %sizes of classes:
    size1 = 120; size2 = 90; size3 = 90;
    %generate points using "mvnrnd" (Random vectors from the multivariate normal distribution) command
    cP1 = mvnrnd(classMean1,classDev1,size1);
    cP2 = mvnrnd(classMean2,classDev2,size2);
    cP3 = mvnrnd(classMean3,classDev3,size3);    
figure(1)
plot(cP1(:,1),cP1(:,2),'.r',cP2(:,1),cP2(:,2),'.g',cP3(:,1),cP3(:,2),'.b','MarkerSize',15);
grid on; axis([-6.5 6.5 -6.5 6.5]);
   
   %sample means:
   sMean1 = mean(cP1);
   fprintf('Sample Mean 1 = %.7f %.7f\n',sMean1);
   sMean2 = mean(cP2);
   fprintf('Sample Mean 2 = %.7f %.7f\n',sMean2);
   sMean3 = mean(cP3);
   fprintf('Sample Mean 3 = %.7f %.7f\n',sMean3);
   
   %sample covariances:
   sCov1 = cov(cP1);
   fprintf('Sample Covariance 1 =\n %.7f %.7f\n %.7f %.7f\n',sCov1);
   sCov2 = cov(cP2);
   fprintf('Sample Covariance 2 = \n %.7f %.7f\n %.7f %.7f\n',sCov2);
   sCov3 = cov(cP3);
   fprintf('Sample Covariance 3 =\n %.7f %.7f\n %.7f %.7f\n',sCov3);

   %class priors:
   sizes = size1 + size2 + size3;
   cPri1 = size1/sizes;
   fprintf('Class Prior 1 = %.1f\n',cPri1);
   cPri2 = size2/sizes;
   fprintf('Class Prior 2 = %.1f\n',cPri2);
   cPri3 = size3/sizes;
   fprintf('Class Prior 3 = %.1f\n',cPri3);
   
   %confuison matrix:
   y1 = ones(120,1);
   y2 = ones(90,1);
   y = [y1; y2*2; y2*3];
   X = [cP1; cP2; cP3];
   
   gmModel = fitgmdist(X,2);
   ypredict = predict(gmModel,X);