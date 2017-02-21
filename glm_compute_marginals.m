function [marg_m, marg_v, H] = glm_compute_marginals(model, data, params)
% function [marg_m, marg_v, H] = glm_compute_marginals(model, data, params)
%
% returns marginal means and variances of approximate posterior evaluated at
% data locations for a generalized linear model
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

H = data.xt';
marg_m = data.xt*params.var.m;
Y = data.xt*params.var.C';
marg_v = sum(Y.*Y, 2);
