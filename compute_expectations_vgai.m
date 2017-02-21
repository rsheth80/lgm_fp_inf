function x = compute_expectations(order, model, data, params, marg_m, marg_v)
% function sum_of_expecs = compute_expectations(0, model, data, params, marg_m, marg_v)
% function rho = compute_expectations(1, model, data, params, marg_m, marg_v)
% function lambda_div_2 = compute_expectations(2, model, data, params, marg_m, marg_v)
%
% returns 0th, 1st, or 2nd order expectation of log pr(y|f) wrt/ 
% N(f|marg_m, marg_v)
% utilizes vgai toolbox likelihood functions interface
%
% assumes:
%   the model structure has fields: 
%       logl_func specifying vgai log likelihood function
%       logl_int set to one of {'numeric', 'analytic'} depending on how the 
%           integrals are computed
%
% known to work with the following vgai likelihood functions:
%   log_siglogit1 (tested with glm, ggpm)
%   exp_log_poisson_loglink (tested with glm)
%   exp_log_normal (tested with glm)
%   exp_log_laplace (tested with glm, sggpm)
%   log_stut (tested with glm)
%   log_ordinal (tested with glm, sggpm)
%   
% NOTE: since there does not appear to be a standardized way to pass likelihood
% parameters around within the vgai toolbox functions (like in gpml toolbox), 
% there is no general way to pass likelihood parameters into this function for
% use by the vgai code in calculating the requested expectations. hence, all 
% likelihood parameters are set to vgai defaults.
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

persistent numInt

% vgai defaults (from init_pot)
if(strcmp(model.logl_int,'numeric') && isempty(numInt))
    numInt.method   = 'quad';
    if(strcmp(func2str(model.logl_func{1}),'log_ordinal'))
        numInt.mv       = 7.5; % non-default value (for numerical stability)
    else
        numInt.mv       = 4.5;
    end;
    numInt.del      = 0.05;
    numInt.z        = -numInt.mv:numInt.del:numInt.mv;
    numInt.pz       = exp(-0.5.*numInt.z.^2)./sqrt(2*pi);
    numInt.zpz      = numInt.z.*numInt.pz;
    numInt.z2m1pz   = (numInt.z.^2-1).*numInt.pz;
end;

if(strcmp(model.logl_int, 'numeric'))
    int_lik = @(f,m,v,om,oc,p) expLogPhi(m,sqrt(v),f,numInt,om,oc,p);
else
    int_lik = @(f,m,v,om,oc,p) feval(f,m,sqrt(v),om,oc,p);
end;

vgai_prms.m = data.yt;  % vgai defaults to using observation as mean of 
                        % likelihood function (where applicable)
vgai_prms.y = data.yt;  % log_siglogit1 and exp_log_poisson_loglink use this
                        % field (and the field appears unused by likelihood/
                        % potential functions of vgai toolbox)
vgai_prms.s = 1;        % needs to be explicitly set in the case of student-t 
                        % likelihood because init_pot sets to s = 1 (desired), 
                        % but dp_stut sets to 1/sqrt(3) (undesired)

switch(order)
case 0
    x = int_lik(model.logl_func{1}, marg_m, marg_v, 0, 0, vgai_prms);
case 1
    [~,x] = int_lik(model.logl_func{1}, marg_m, marg_v, 1, 0, vgai_prms);
case 2
    [~,~,x] = int_lik(model.logl_func{1}, marg_m, marg_v, 0, 1, vgai_prms);
end;
