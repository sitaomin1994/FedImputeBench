#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# imputation
#./scripts/run_scenario_cpu.sh vehicle_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter simple,em,linear_ice local,fedavg -1
#./scripts/run_scenario_cpu.sh vehicle_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter missforest local,fedtree -1
#./scripts/run_scenario_cpu.sh school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter simple,em,linear_ice local,fedavg -1
#./scripts/run_scenario_cpu.sh school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter missforest local,fedtree -1
#./scripts/run_scenario_cpu.sh vehicle_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter gain local,fedavg,fedavg_ft,fedprox -1
#./scripts/run_scenario_cpu.sh school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter gain local,fedavg,fedavg_ft,fedprox -1
#./scripts/run_scenario_cpu.sh vehicle_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter miwae local,fedavg,fedavg_ft,fedprox -1
#./scripts/run_scenario_cpu.sh school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter miwae local,fedavg,fedavg_ft,fedprox -1

# evaluation
./scripts/run_eval.sh vehicle_np,school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
simple,em,linear_ice local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle_np,school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle_np,school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
./scripts/run_eval.sh vehicle_np,school_np natural-partition mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#miwae local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#gain local,fedavg,fedavg_ft,fedprox nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#missforest local,fedtree nn -1 fed_imp_pc2
#./scripts/run_eval_fed.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-homog,mnar2-heter \
#simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2