% assumes base dir is two levels up from current
basedir = fullfile('..','..');
run(fullfile(basedir,'gpml-matlab-v3.4-2013-11-11','gpml_startup'));
addpath(fullfile(basedir,'minFunc_2012'));
addpath(fullfile(basedir,'minFunc_2012','minFunc'));
addpath(fullfile(basedir,'minFunc_2012','minFunc','compiled'));
addpath(fullfile(basedir,'vgai_1.1'));
addpath(fullfile(basedir,'vgai_1.1','utils'));
addpath(genpath(fullfile(basedir,'vgai_1.1','potfuns')));
addpath(fullfile(pwd,'util'));
