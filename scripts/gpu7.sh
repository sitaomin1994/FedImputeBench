#!/bin/bash

./scripts/run_eval.sh codrna niid-f1 mnar2-homog \
miwae local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f1 mnar2-homog \
gain local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f1 mnar2-homog \
missforest local,fedtree,central nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna niid-f1 mnar2-homog \
simple,em,linear_ice local,fedavg,central nn -1 fed_imp_pc2