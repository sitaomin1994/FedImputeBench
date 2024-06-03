#!/bin/bash


##./scripts/run_eval.sh school_pca niid-f2 mnar2-homog \
##miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval.sh school_pca niid-f2 mnar2-homog \
#gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
##./scripts/run_eval.sh school_pca niid-f2 mnar2-homog \
##missforest local,fedtree nn -1 fed_imp_pc2
##./scripts/run_eval.sh school_pca niid-f2 mnar2-homog \
##simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 mnar2-homog \
gain local,fedavg,fedavg_ft,fedprox nn 10 fed_imp_pc2