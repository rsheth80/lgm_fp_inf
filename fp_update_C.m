function C = fp_update_C(model, data, params, calcs)
% function C = fp_update_C(model, data, params, calcs)
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

if(nargin<4)
    [~, L] = feval(model.prior_func{:}, model, data, params);
    icov = L\((L')\eye(model.D));
else
    icov = calcs.dvlb_inv_s;
end;

[marg_m, marg_v, H] = feval(model.marg_func{:}, model, data, params);
lambda_div_2 = feval(model.expec_func{:}, 2, model, data, params, ...
                    marg_m, marg_v);
C = chol((icov + H*bsxfun(@times, H', -2*lambda_div_2))\eye(model.D));
