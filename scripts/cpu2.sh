#!/bin/bash

# dvisits
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-homog missforest \
local,fedtree linear -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-homog simple,em,linear_ice \
local,fedavg linear -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-homog gain \
local,fedavg linear -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-homog miwae \
local,fedavg linear -1 fed_imp_pc2

# school pca
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-t1,niid-t2 mar-homog missforest \
local,fedtree linear -1 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-t1,niid-t2 mar-homog simple,em,linear_ice \
local,fedavg linear -1 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-t1,niid-t2 mar-homog gain \
local,fedavg linear -1 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-t1,niid-t2 mar-homog miwae \
local,fedavg linear -1 fed_imp_pc2
