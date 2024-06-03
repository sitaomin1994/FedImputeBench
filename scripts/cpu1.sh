#!/bin/bash

./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mcar,mar-homog,mnar2-homog \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mcar,mar-homog,mnar2-homog \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mcar,mar-homog,mnar-homog \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mcar,mar-homog,mnar-homog \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2