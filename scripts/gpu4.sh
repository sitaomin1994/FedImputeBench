#!/bin/bash

./scripts/run_eval.sh school_pca niid-f1 mar-homog \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mar-homog \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mar-homog \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mar-homog \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2