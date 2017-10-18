% Example script for basic Kohonen map algorithm.
clear all
close all

data = dlmread('data.txt'); % read in data
labels = dlmread('labels.txt'); % read in labels

name = 'Matthias Chinyen Tsai'; % REPLACE BY YOUR OWN NAME
targetdigits = name2digits(name); % assign the four digits that should be used

digitstoremove = setdiff(0:9,targetdigits); % the other 6 digits are removed from the data set.
for i=1:6
    data(labels==digitstoremove(i),:) = [];
    labels(labels==digitstoremove(i)) = [];
end

dim = 28*28; % dimension of the images
range = 255; % input range of the images ([0, 255])
[dy, dx]=size(data);

% set the size of the Kohonen map. In this case it will be 6 X 6
sizeK=6;
NrNeurons=sizeK^2; %set the total number of neurons

%set the width of the neighborhood via the width of the gaussian that
%describes it
% sigma constant
sigma=3;

% sigma min and max when the neighborhood width is decreasing over time :
% to do it, the section " decrease of the neighborhood width with in a
% quadratic/linear fashion" must be uncomment
sigmamax = 6;
sigmamin = 1;



%initialise the centers randomly
centers=rand(NrNeurons,dim)*range;

% build a neighborhood matrix
neighbor = reshape(1:NrNeurons,sizeK,sizeK);

% YOU HAVE TO SET A LEARNING RATE HERE:

eta = 0.005;
%set the maximal iteration count
tmax=5000; % this might or might not work; use your own convergence criterion

%set the random order in which the datapoints should be presented
iR=mod(randperm(tmax),dy)+1;
deltaomega=zeros(tmax,1);
for t=1:tmax
    i=iR(t);
    old=centers;
    centers=som_step(centers,data(i,:),neighbor,eta,sigma);
    
     
    %% abs : difference between old and new centers
    deltaomega(t) = sum(sum(abs(old-centers)));
    
    %% decrease of the neighborhood width in a linear fashion
    %     sigma = sigmamax-t*(sigmamax-sigmamin)/tmax;
    
    %% decrease of the neighborhood width in a quadratic fashion
    %sigma = sigmamin + sigmamax*(t/tmax-1)^2;
end

%% Batch algorithm for one step in order to add a supplementary criterion for the determination of the convergence
batchomega = zeros(sizeK*sizeK,dim);
for i=1:dy
    o=som_step(centers,data(i,:),neighbor,eta,sigma);
    batchomega = batchomega + centers-o;
end
batchomega = batchomega/dy;
b=0;
for i=1:NrNeurons
    b=b+norm(batchomega(i,:));
end
batch = b/eta;

errors = zeros(4,1);
numberlabel = errors;
neuronlabel = zeros(sizeK^2,1);

%digits assignement
for n=1:sizeK^2
    errors = zeros(4,1);
    numberlabel = errors;
 
    for d=1:length(data)
        for m=1:4
            if(labels(d)== targetdigits(m))
               numberlabel(m) = numberlabel(m) + 1; 
               errors(m) = errors(m) + sum(abs(data(d,:) - centers(n,:))); 
               %errors(m) = errors(m) + norm(data(d,:) - centers(n,:)); 
            end
        end
    end
   
    for m=1:4
        errors(m) = errors(m)/numberlabel(m);
    end
    [a,b]=min(errors);
    neuronlabel(n)=targetdigits(b);
end

%% for visualization, you can use this:
figure('units','normalized','outerposition',[0 0 1 1]);
for i=1:sizeK^2
    subplot(sizeK,sizeK,i);
    imagesc(reshape(centers(i,:),28,28)'); colormap gray;
    title(neuronlabel(i));
    axis off
end

%% plot delta omega

figure(2)
 title('Convergence criteria')
 plot(1:tmax,smooth(deltaomega,40)/eta,'b');  %smooth = remove noise
%plot(1:tmax,(deltaomega)/eta,'b');