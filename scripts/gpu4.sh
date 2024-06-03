#!/bin/bash

./scripts/run_eval.sh codrna niid-f2 mar-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f2 mar-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f2 mar-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f2 mar-heter \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2