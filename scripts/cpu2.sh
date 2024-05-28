# codrna
#./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 simple,em,linear_ice local,fedavg,central linear -1
#./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 missforest local,fedtree,central linear -1
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 miwae local,fedavg,fedprox,fedavg_ft linear 10
./scripts/run_eval.sh codrna iid-even,iid-uneven,niid-f1,niid-f2 gain local,fedavg,fedprox,fedavg_ft linear 10

#./scripts/run_eval.sh california iid-even,iid-uneven,niid-f1,niid-f2 simple,em,linear_ice local,fedavg,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-f1,niid-f2 missforest local,fedtree,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-f1,niid-f2 miwae,gain local,fedavg,fedprox,fedavg_ft,central nn -1