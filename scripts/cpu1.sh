#!/bin/bash

./scripts/run_eval_fed.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mnar2-heter,mar-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mnar2-heter,mar-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mnar2-heter,mar-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval_fed.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mnar2-heter,mar-heter \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2