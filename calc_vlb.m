function [vlb, dvlb, dvlb_inv_v, dvlb_inv_s, dvlb_lik] = ...
    calc_vlb(model, data, params, calcs)
% function [vlb, dvlb, dvlb_inv_v, dvlb_inv_s, dvlb_lik] = ...
%   calc_vlb(model, data, params, calcs)
%
% vlb           = -KL[N(m,V) || N(mu,S)] + sum_i E_{q_i(f_i)}[log p(y_i|f_i), 
%                   where mu, S, are prior mean and cov, and q_i = N(m_i,v_i)
% dvlb          = [dvlb/dm; trilv(dvlb/dV)]
%
% the prior mean and cov are computed with model.prior_func
% the marginals (m_i,v_i) are computed with model.marg_func
% the sum of expectations is computed with model.expec_func
%
% calcs is an optional input structure that can contain:
%     dvlb_inv_v    = inv(V)
%     dvlb_inv_s    = inv(S)
%     dvlb_lik      = H*diag(lambda)*H'
% if calcs is supplied then fields that exist in calcs will not be recomputed
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

if(nargout>1)
    DOSECOND = 1;
else
    DOSECOND = 0;
end;

if(nargin>3 && ~isempty(calcs))
    ISCALCS = 1;
else
    ISCALCS = 0;
end;

logdetchol = @(x) 2*sum(log(abs(diag(x))));
C = params.var.C;
m = params.var.m;
D = model.D;

if(DOSECOND)
    dvlb = zeros(D + 0.5*D*(D+1), 1);
end;

% vlb KL-div terms
[mu, L] = feval(model.prior_func{:}, model, data, params);
x = (L')\(m - mu);
Y = (L')\(C');
Y = Y.*Y;

vlb = 0.5*(D + logdetchol(C) - logdetchol(L) - sum(x.^2) - sum(Y(:)));

% vlb lik terms
[marg_m, marg_v, H] = feval(model.marg_func{:}, model, data, params);
vlb = vlb + feval(model.expec_func{:}, 0, model, data, params, ...
                    marg_m, marg_v);

if(DOSECOND)

    % dvlb KL-div terms
    dvlb(1:D) = -L\x;
    if(ISCALCS && isfield(calcs,'dvlb_inv_v') && ~isempty(calcs.dvlb_inv_v))
        dvlb_inv_v = calcs.dvlb_inv_v;
    else
        dvlb_inv_v = C\(C'\eye(D));
    end;
    if(ISCALCS && isfield(calcs,'dvlb_inv_s') && ~isempty(calcs.dvlb_inv_s))
        dvlb_inv_s = calcs.dvlb_inv_s;
    else
        dvlb_inv_s = L\(L'\eye(D));
    end;
    dvlb((D+1):end) = 0.5*full_to_trilv(dvlb_inv_v - dvlb_inv_s);

    % dvlb lik terms
    rho = feval(model.expec_func{:}, 1, model, data, params, marg_m, marg_v);
    dvlb(1:D) = dvlb(1:D) + H*rho;
    lambda_div_2 = feval(model.expec_func{:}, 2, model, data, params, ...
                        marg_m, marg_v);
    if(ISCALCS && isfield(calcs,'dvlb_lik') && ~isempty(calcs.dvlb_lik))
        dvlb_lik = calcs.dvlb_lik;
    else
        dvlb_lik = H*bsxfun(@times, H', lambda_div_2);
    end;
    dvlb((D+1):end) = dvlb((D+1):end) + full_to_trilv(dvlb_lik);

end;
