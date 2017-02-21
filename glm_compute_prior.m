function [m, L, K] = glm_compute_prior(model, data, params)
% function [m, L, K] = glm_compute_prior(model, data, params)
%
% returns prior mean and cholesky factor of prior covariance (and optionally
% prior covariance) for a generalized linear model
%
% will check params for fields m_prior, chol_prior, and cov_prior first
% defaults to zero vector and identity matrix
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


if(isfield(params, 'm_prior'))
    m = params.m_prior;
else
    m = zeros(model.D, 1);
end;

if(isfield(params, 'chol_prior'))
    L = params.chol_prior;
elseif(~isfield(params, 'chol_prior') && isfield(params, 'cov_prior'))
    L = chol(params.cov_prior);
else
    L = speye(model.D);
end;

if(nargout>2)
    if(isfield(params, 'chol_prior') && ~isfield(params, 'cov_prior'))
        K = params.chol_prior'*params.chol_prior;
    else
        K = speye(model.D);
    end;
end;
