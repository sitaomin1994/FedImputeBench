# california
#./scripts/run_scenario_cpu.sh school_pca niid-f1,niid-f2 mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
#simple,em,linear_ice central -1
./scripts/run_scenario_cpu.sh school_pca niid-f1,niid-f2 mcar,mar-homog,mar-heter,mnar1-homog,mnar1-heter,mnar2-homog,mnar2-heter \
missforest central -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central nn -1 fed_imp_pc
# ./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central linear,nn -1 fed_imp_pc
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 miwae local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
#/scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 gain local,fedavg,fedprox,fedavg_ft,central linear,nn -1 fed_imp_pc
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 simple,em,linear_ice local,fedavg,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 missforest local,fedtree,central nn -1
#./scripts/run_eval.sh california iid-even,iid-uneven,niid-t1,niid-t2 miwae,gain local,fedavg,fedprox,fedavg_ft,central nn -1
