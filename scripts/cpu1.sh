# california
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central linear -1
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central linear -1
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 miwae,gain local,fedavg,fedprox,fedavg_ft,central linear -1

./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central nn -1
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central nn -1
./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 miwae,gain local,fedavg,fedprox,fedavg_ft,central nn -1