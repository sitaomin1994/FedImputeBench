#!/bin/bash


./scripts/run_eval.sh codrna iid-uneven mar-heter \
miwae local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
gain local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
missforest local,fedtree,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-uneven mar-heter \
simple,em,linear_ice local,fedavg,central nn -1 fed_imp_pc2