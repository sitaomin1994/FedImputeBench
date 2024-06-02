#!/bin/bash

./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mcar \
miwae local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mcar \
gain local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mcar \
missforest local,fedtree,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mcar \
simple,em,linear_ice local,fedavg,central nn -1 fed_imp_pc2
