#!/bin/bash

./scripts/run_eval.sh codrna iid-even mar-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even mar-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even mar-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even mar-heter \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2


