#!/bin/bash
# ./scripts/run_eval.sh dvisits iid-uneven mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
#./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter simple,em,linear_ice,miwae,gain local,fedavg nn -1 fed_imp_pc2
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter simple,em,linear_ice,miwae,gain local,fedavg nn -1 fed_imp_pc2
./scripts/run_eval.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2 mar-heter,mnar2-heter missforest local,fedtree nn -1 fed_imp_pc2
# ./scripts/run_eval.sh codrna niid-f1,niid-f2 mar-heter gain local,fedavg nn 10 fed_imp_pc2
# ./scripts/run_eval.sh codrna iid-even,iid-uneven mnar2-heter,mar-heter miwae local,fedavg nn 10 fed_imp_pc2
# ./scripts/run_eval.sh codrna iid-even,iid-uneven mar-heter gain local,fedavg nn 10 fed_imp_pc2