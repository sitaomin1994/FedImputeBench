#!/bin/bash


./scripts/run_eval.sh codrna iid-uneven mar-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2