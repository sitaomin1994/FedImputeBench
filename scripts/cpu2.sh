#!/bin/bash

./scripts/run_eval_fed.sh california iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mnar2-homog \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh california iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mnar2-homog \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh california iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mnar2-homog \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh california iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mnar2-homog \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2
