#!/bin/bash

# school pca
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2 mcar,mar-heter,mar-homog,mnar2-heter,mar-heter \
simple,linear_ice,em local,fedavg linear 10 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2 mcar,mar-heter,mar-homog,mnar2-heter,mar-heter \
missforest local,fedtree linear 10 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2 mcar,mar-heter,mar-homog,mnar2-heter,mar-heter \
miwae local,fedavg linear 10 fed_imp_pc2
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2 mcar,mar-heter,mar-homog,mnar2-heter,mar-heter \
gain local,fedavg linear 10 fed_imp_pc2
