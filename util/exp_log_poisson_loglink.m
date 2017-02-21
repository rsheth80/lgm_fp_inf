function [sumIn dIdm dIdsig2] = exp_log_poisson_loglink(mn,sn,varargin)
% This function analytically evaluates the Gaussian expectation of the
% log of the Poisson pdf with log link function relating the rate parameter to
% the variational Gaussian approximation; the expectation is taken wrt a 
% Gaussian with mean mn(n) and standard deviation sn(n)
%     I_n := <log p(x|m,s)>_N(x|mn(n),sn(n))
%     sumIn = sum_n I_n
% We also evaluate the expectations derivatives wrt mn and sn^2.
% 
% the Poisson pdf is parameterised
% p(x|y) = 1/y! * exp(-exp(x)) * exp(x*y)

% varargin={optm,optc,params}
% optm, optc booleans specifying if we compute derivative wrt mn and sn^2
% mn - Nx1 vector of Gaussian means
% sn - Nx1 vector of Gaussian stds.

if numel(varargin)==0
  optm=1; optc=1;
  params=struct;
elseif numel(varargin)==1
  optm=varargin{1};
  optc=1;
  params=struct;
elseif numel(varargin)==2
  optm=varargin{1};
  optc=varargin{2};
  params=struct;
elseif numel(varargin)==3
  optm=varargin{1};
  optc=varargin{2};
  params=varargin{3};
end
  
y = params.y;
a = bsxfun(@plus,mn,0.5*sn.^2);
expa = exp(a);

% sum_n <log phi(mn+z*sn)>
sumIn = bsxfun(@plus,-expa,bsxfun(@times,mn,y)) - gammaln(y+1);
sumIn = sum(sumIn);

% evaluate d/d(mn) <log phi(mn+z*sn)>_N(z|0,1) with 
if optm
    dIdm = bsxfun(@plus,-expa,y);
else
  dIdm=0;
end

% evaluate d/d(sn^2) <log phi(mn+z*sn)>_N(z|0,1) where
if optc
    %dIdsig2 = bsxfun(@times,-expa,sn);
    dIdsig2 = -expa/2;
else
  dIdsig2=0;
end
