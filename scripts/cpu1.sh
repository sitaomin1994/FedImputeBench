#!/bin/bash
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central linear,nn -1 fed_imp_pc
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 miwae local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
# ./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 gain local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 simple,em,linear_ice local,fedavg,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 missforest local,tree,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 miwae local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 gain local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 miwae,gain local,fedavg,fedprox,fedavg_ft,central nn -1
