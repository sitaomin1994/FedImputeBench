#!/bin/bash


./scripts/run_eval.sh school_pca niid-f2 mnar2-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mnar2-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mnar2-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mnar2-heter \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2