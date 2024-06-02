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

#./scripts/run_eval.sh codrna niid-f1,niid-f2 mar-heter,mnar2-heter gain local,fedavg nn 10 fed_imp_pc2
#./scripts/run_eval.sh codrna niid-f1,niid-f2 mar-heter,mnar2-heter simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2
#./scripts/run_eval.sh codrna niid-f1,niid-f2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter simple,em,linear_ice,miwae,gain local,fedavg nn -1 fed_imp_pc2
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2