#!/bin/bash

./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 mcar \
miwae local,fedavg,fedavg_ft,fedprox nn 8 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 mar-homog \
gain local,fedavg,fedavg_ft,fedprox nn 10 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 mar-homog \
missforest local,fedtree nn -1 fed_imp_pc2