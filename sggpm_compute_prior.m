function [m, L, K, m_train, Kii, Knm] = sggpm_compute_prior(model, data, params)
% function [m_ind, L_ind, K_ind, m_train, Kii, Knm] = sggpm_compute_prior(model, data, params)
%
% returns prior mean and cholesky factor of prior covariance (and optionally
% prior covariance) for a sparse generalized GP model
%
% assumes:
%   the model structure has fields mean_func and cov_func
%   the params struct has fields hyp.mean, hyp.cov, and var.xinducing
%
% will check params for fields m_prior, chol_prior, and cov_prior first
%
% NOTE: does *not* perform verification that chol_prior'*chol_prior is equal to 
% cov_prior if they both exist in params
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

keps = 1e-9;

if(isfield(params, 'm_prior'))
    m = params.m_prior;
else
    m = feval(model.mean_func{:}, params.hyp.mean, params.var.xinducing);
end;

if(isfield(params, 'chol_prior'))
    L = params.chol_prior;
    K = params.cov_prior;
else
    K = feval(model.cov_func{:}, params.hyp.cov, params.var.xinducing) ...
            + keps*eye(model.D);
    L = chol(K);
end;

if(isfield(params, 'm_train'))
    m_train = params.m_train;
else
    m_train = feval(model.mean_func{:}, params.hyp.mean, data.xt);
end;

if(isfield(params, 'Kii'))
    Kii = params.Kii;
else
    Kii = feval(model.cov_func{:}, params.hyp.cov, data.xt, 'diag');
end;

if(isfield(params,'Knm'))
    Knm = params.Knm;
else
    Knm = feval(model.cov_func{:}, params.hyp.cov, data.xt, ...
            params.var.xinducing);
end;
