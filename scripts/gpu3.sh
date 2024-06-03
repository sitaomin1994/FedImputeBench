#!/bin/bash

./scripts/run_eval.sh school_pca niid-f1 mcar \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mcar \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mcar \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh school_pca niid-f1 mcar \
simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2