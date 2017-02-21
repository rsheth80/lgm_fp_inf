function X = trilv_to_full(v)
% function X = trilv_to_full(v)
%
% converts [0.5*n*(n+1)]-length vector of lower triangular part of X to full sym
% matrix
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

nel = length(v);
D = 0.5*(-1+sqrt(1+8*nel));
ix0 = tril(reshape(1:D^2, D, D));

% an index set that will convert a vector holding the lower triangular part of a
% sym matrix in to a full symm matrix
ix = zeros(D);
ix(ix0(ix0>0)) = 1:nel;
ixd = diag(ix);
ixdtrue = diag(reshape(1:D^2,D,D));
ix = ix + ix';
ix(ixdtrue) = ixd;
ix_trilv_to_full = ix;

X = reshape(v(ix_trilv_to_full), D, D);
