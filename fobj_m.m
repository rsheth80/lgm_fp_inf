function [f, df, dff] = fobj_m(m, model, data, params, calcs)
% function [f, df, dff] = fobj_m(m, model, data, params, calcs)
%
% objective function (-VLB) for m optimization
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

params.var.m = m;
if(nargin<5)
    [vlb, dvlb, ~, dvlb_inv_s, dvlb_lik] = calc_vlb(model, data, params);
else
    [vlb, dvlb, ~, ~, dvlb_lik] = calc_vlb(model, data, params, calcs);
    dvlb_inv_s = calcs.dvlb_inv_s;
end;
f = -vlb;
df = -dvlb(1:model.D);
dff = dvlb_inv_s - dvlb_lik;
