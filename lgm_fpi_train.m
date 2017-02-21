function [pout, exflag, trace] = lgm_fpi_train(model, data, params, opts)
% function [pout, excond, trace] = lgm_fpi_train(model, data, params, opts)
%
% trains an LGM model with the incremental fixed point (FPi) method which 
% consists of one step of Newton optimization in the mean and one fixed point
% update in the cov using the previous cov
%
% check README file for description of (model, data, params)
% opts is an optional structure specifying optimization settings:
%   optTol      :   threshold 1st-order optimality condition (default: 1e-5)
%   optProg     :   minimum change in objective function value required to 
%                   continue (default: 1e-9) 
%   MaxIter     :   maximum number of outer-loop iterations (default: 100)
%
% pout is output (trained) parameter set
% excond specifies why the optimization stopped: converged (1), exceeded max. 
% number of iterations (0), could not take a step (2)
% trace is a structure that holds, per iteration:
%   fs          :   objective function value
%   ts          :   cpu time (sec)
%   tws         :   wall clock time (sec)
%
% NOTE: the objective function is equal to -VLB
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default optimization settings (matched to minFunc)
tolopt = 1e-5;
tolprog = 1e-9;
maxiter = 100;

% read user settings, if any
if(nargin>3 && ~isempty(opts))
    if(isfield(opts,'optTol'))
        tolopt = opts.optTol;
    end;
    if(isfield(opts,'optProg'))
        tolprog = opts.optProg;
    end;
    if(isfield(opts,'MaxIter'))
        maxiter = opts.MaxIter;
    end;
end;

% start timing of initialization
t0 = cputime;
tw0 = tic;

% inner-loop options
optsm.method = 'newton';
optsm.display = 'off';
optsm.maxiter = 1;

% optim init
pout = params;
iter = 0;
abs_f_change = inf;
calcs = struct;
[vlb, dvlb, calcs.dvlb_inv_v, calcs.dvlb_inv_s] = calc_vlb(model, data, pout);
f = -vlb;
first_order_opt = max(abs(dvlb));

% record timing of initialization
tw0 = toc(tw0);
tw = tw0;
t0 = cputime-t0;
t = t0;

% save data
trace.fs = [f];
trace.ts = [t];
trace.tws = [tw];

% tmp vars
this_t = 0; this_tw = 0; old_f = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      LOOP       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% outer optim loop
fprintf('Iteration: %6i;  f: %4.6e\r', iter, f);
while(first_order_opt > tolopt && iter < maxiter && abs_f_change > tolprog)

    old_f = f;

    % start timing of outer loop 
    this_t = cputime;
    this_tw = tic;

    % do one step in m with Newton's method
    pout.var.m = minFunc(@fobj_m, pout.var.m, optsm, model, data, pout, calcs);

    % do a fixed point update on V
    pout.var.C = fp_update_C(model, data, pout, calcs);

    % C changed -> inv(V) needs to be recomputed
    calcs = rmfield(calcs, 'dvlb_inv_v');

    % calculate values for stopping tests
    [vlb, dvlb, calcs.dvlb_inv_v] = calc_vlb(model, data, pout, calcs);
    f = -vlb;
    abs_f_change = abs(old_f - f);
    first_order_opt = max(abs(dvlb));

    % record timing of outer loop 
    this_tw = toc(this_tw);
    this_t = cputime - this_t;

    % save data
    t = t + this_t;
    tw = tw + this_tw;
    trace.fs = [trace.fs;f];
    trace.ts = [trace.ts;t];
    trace.tws = [trace.tws;tw];

    % increment outer-loop counter
    iter = iter + 1;

    fprintf('Iteration: %6i;  f: %4.6e (%4.6e 1st-order opt)\r', ...
        iter, f, first_order_opt );
end;
fprintf('Iteration: %6i;  f: %4.6e (%4.6e 1st-order opt)\n', ...
    iter, f, first_order_opt );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      EXIT       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(first_order_opt <= tolopt)
    exflag = 1; % converged
elseif(iter >= maxiter)
    exflag = 0; % exceeded max iterations
else
    exflag = 2; % could not take a step
end;

fprintf('Exit condition: %d\n', exflag);
