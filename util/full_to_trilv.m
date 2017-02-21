function v = full_to_trilv(X)
% function v = full_to_trilv(X)
%
% converts full sym nxn matrix to [0.5*n*(n+1)]-length vector of lower 
% triangular part of X
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

D = size(X,1);

% an index set that will convert a full symm matrix into a vector holding the 
% lower triangular part
ix0 = tril(reshape(1:D^2, D, D));
ix_full_to_trilv = ix0(ix0>0);

v = X(ix_full_to_trilv);
