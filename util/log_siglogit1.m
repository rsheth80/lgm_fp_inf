function z=log_siglogit1(x,varargin)
% Function that evaluates the log of pmf for the logistic sigmoid.
%
% p(c=1|x)=1/(1+exp(-x))
% logistic sigmoid has the symmetry property that 
% p(c=-1|x)=1-p(c=1|x)=p(c=1|-x)
% 
% so log p(c=1|x)=-log(1+exp(-x))
% when x is large and +ve log p(c=1|x) \approx 0
% when x is large and -ve log p(c=1|x) \approx x
% large +ve or -ve values are assumed when x>15 or -x>15
%
% this version is based on the original log_siglogit function included in vgai 
% and uses the label values input through params.y. this function was written 
% for uniformity with the vgai implementation of the poisson likelihood 
% [RS 5/30/15]

y=varargin{1}.y;
x=bsxfun(@times,x,y);
mv=15;
smallIdx=x<-mv;
bigIdx=x>mv;
sigIdx=~(smallIdx | bigIdx);
z=zeros(size(x));
z(smallIdx)=x(smallIdx);
z(sigIdx)=-log(1+exp(-x(sigIdx)));
