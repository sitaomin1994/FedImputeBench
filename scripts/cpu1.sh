#!/bin/bash
# linear_ice
#./scripts/run_scenario_cpu2.sh codrna iid-even,iid-uneven,niid-f1,niid-f2
#./scripts/run_scenario_cpu2.sh california iid-even,iid-uneven,niid-t1,niid-t2
#./scripts/run_scenario_cpu2.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2
#./scripts/run_scenario_cpu2.sh hhip iid-even,iid-uneven,niid-t1,niid-t2
#./scripts/run_scenario_cpu2.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2
#./scripts/run_scenario_cpu2.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2

# evaluation linear_ice
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "linear_ice"
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh hhip iid-even,iid-uneven,niid-t1,niid-t2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2 linear_ice local,fedavg,central linear,nn -1 fed_imp_pc

# evaluation simple,em
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "em.simple"
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-t1,niid-t2 simple,em local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh hhip iid-even,iid-uneven,niid-f1,niid-f2 simple,em local,fedavg,central linear,nn -1 fed_imp_pc

# evaluation missforest
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "missforest"
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh hhip iid-even,iid-uneven,niid-f1,niid-f2 missforest local,fedtree,central linear,nn -1 fed_imp_pc

# evaluation gain
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "gain"
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-t1,niid-t2 gain local,fedavg,fedavg_ft,fedprox,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh hhip iid-even,iid-uneven,niid-f1,niid-f2 gain local,fedavg,fedavg_ft,fedprox,central linear,nn -1 fed_imp_pc

# evaluation miwae
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "miwae"
echo "================================================================================================================================="
echo "================================================================================================================================="
echo "================================================================================================================================="
./scripts/run_eval.sh vehicle iid-even,iid-uneven,niid-t1,niid-t2 gain local,fedavg,fedavg_ft,fedprox,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh hhip iid-even,iid-uneven,niid-f1,niid-f2 miwae local,fedavg,fedavg_ft,fedprox,central linear,nn -1 fed_imp_pc