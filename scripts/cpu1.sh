#!/bin/bash

./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mnar2-heter \
miwae local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mnar2-heter \
gain local,fedavg,fedavg_ft,fedprox,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mnar2-heter \
missforest local,fedtree,central nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mnar2-heter \
simple,em,linear_ice local,fedavg,central nn -1 fed_imp_pc2