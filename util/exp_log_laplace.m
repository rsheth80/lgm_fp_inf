function [sumIn dIdm dIdsig2]...
  =exp_log_laplace(mn,sn,varargin)
% This function analytically evaluates the Gaussian expectation of the
% log of the Laplace pdf where the expectation is taken wrt a Gaussian 
% with mean mn(n) and standard deviation sn(n)
%     I_n := <log p(x|m,s)>_N(x|mn(n),sn(n))
%     sumIn = sum_n I_n
% We also evaluate the expectations derivatives wrt mn and sn^2.
% 
% the Laplace pdf is parameterised
% p(x|m,s) = (1/(2*s))*exp(-abs(r))
% where r=abs(x-m)./s;

% varargin={optm,optc,params}
% optm, optc booleans specifying if we compute derivative wrt mn and sn^2
% mn - Nx1 vector of Gaussian means
% sn - Nx1 vector of Gaussian stds.
%
% dIdsig2 corrected [RS 5/30/15]

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
  
% get default params
params=dp_laplace(params);

mn=bsxfun(@plus,mn,-params.m);
a=mn./sn;

uga=exp(-0.5.*(a.^2));
iga=normcdf(-a);

% sum_n <log phi(mn+z*sn)>
sumIn=(sqrt(2/pi).*sn.*uga+mn.*(1-2.*iga))./params.s;
sumIn=bsxfun(@plus,sumIn,log(2*params.s));
sumIn=-sum(sumIn,1);

% evaluate d/d(mn) <log phi(mn+z*sn)>_N(z|0,1) with 
% phi(x):=exp(-|x|/s)
if optm
  dIdm=(2.*iga-ones(size(mn)))./params.s;
else
  dIdm=0;
end

% evaluate d/d(sn^2) <log phi(mn+z*sn)>_N(z|0,1) where
% phi(x):=exp(-|x|/s)
if optc
  % d/ds <|w|>_N(w|m,s^2)
  dIdsig2=(1./sqrt(2*pi)).*(uga./sn).*(1+a.^2)...
    -(a.^2)./sn.*normpdf(a);
%    -(1.*a.^2).*uga./(sn.*sqrt(2*pi));
  % d/ds -<|w|/s>
  dIdsig2=-dIdsig2./params.s;
else
  dIdsig2=0;
end

