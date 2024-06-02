#!/bin/bash

./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter miwae local,fedavg nn 10 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter gain local,fedavg nn 10 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter simple,em,linear_ice local,fedavg nn -1 fed_imp_pc2
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter simple,em,linear_ice,miwae,gain local,fedavg nn -1 fed_imp_pc2
#./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2