function z = log_ordinal(x,varargin)
%
% uses likCumLog to compute log probabalities at input locations
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

% fixed likelihood hyperparameters 
n = 5;
delta = 2;
slope = 50;
hyp.lik = [-1;log(ones(n-2,1)*delta/(n-2));log(slope)];

y = varargin{1}.y;
z = zeros(size(x));
for i = 1:size(x,2)
    z(:,i) = likCumLog(n, hyp.lik, y, x(:,i), [], 'infLaplace');
end;
