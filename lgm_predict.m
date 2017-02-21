function datar = lgm_predict(model,data,params,xs,ys)
% function predictions = lgm_predict(model,data,params,xs)
% function predictions = lgm_predict(model,data,params,xs,ys)
%
% xs is Ntest x D test locations
% predictions is a structure:
%   x:          [Ntest x D]     equal to xs
%   f_mean:     [Ntest x 1]     posterior latent mean evaluated at xs
%   f_var:      [Ntest x 1]     posterior latent variance evaluated at xs
%   y_mean:     [Ntest x 1]     output mean evaluated at xs
%   y_var:      [Ntest x 1]     output variance evaluated at xs
%   y_mode:     [Ntest x 1]     output mode evaluated at xs (for ordinal, 
%                               logistic likelihoods)
%
% if test outputs, ys, are supplied, then predictions will have the 
% additional field
%   lp:         [Ntest x 1]     log probability
%
% uses GPML toolbox to compute predictions
%
% Copyright (C) 2017  Rishit Sheth

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

data_test.xt = xs;
[marg_m, marg_v] = feval(model.marg_func{:}, model, data_test, params);

% response mean/var
if(nargin<5)
    [~,y_pred_mean,y_pred_var]=feval(model.lik_func{:},params.hyp.lik,[], ...
                                    marg_m,marg_v);
else
    [lp,y_pred_mean,y_pred_var]=feval(model.lik_func{:},params.hyp.lik,ys, ...
                                    marg_m,marg_v);
end;

% compute modes for ordinal and logistic likelihoods
y_mode_flag = true;
switch(func2str(model.lik_func{1}))
case 'likCumLog'
    y_pred_mode = zeros(size(xs,1),1);
    Lc = model.lik_func{2};
    lp = zeros(size(xs,1),1);
    lpi = zeros(1,Lc);
    for i = 1:size(xs,1)
        lpi = feval(model.lik_func{:},params.hyp.lik,1:Lc,marg_m(i),marg_v(i));
        [lp(i),y_pred_mode(i)] = max(lpi);
    end;
case 'likLogistic'
    n = length(y_pred_mean);
    y_pred_mode = -1*ones(n,1);
    if(nargin<5)
        lp = feval(model.lik_func{:},params.hyp.lik,zeros(n,1),marg_m,marg_v);
    end;
    y_pred_mode(lp>log(0.5)) = 1;
otherwise
    y_mode_flag = false;
end;

datar = struct;
datar.x = xs;
datar.f_mean=marg_m;
datar.f_var=marg_v;
datar.y_mean=y_pred_mean;
datar.y_var=y_pred_var;
if(y_mode_flag)
    datar.y_mode = y_pred_mode;
end;
if(nargin>4)
    datar.lp = lp;
end;

