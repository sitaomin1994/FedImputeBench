#!/bin/bash

./scripts/run_eval.sh school_pca niid-f2 mcar \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mcar \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mcar \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f2 mcar \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2