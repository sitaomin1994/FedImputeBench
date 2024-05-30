#!/bin/bash
# 14
./scripts/run_scenario_gpu.sh hhip iid-even mcar,mar-homog,mnar2-homog fedavg_ft 8
./scripts/run_scenario_gpu.sh hhip iid-uneven mcar,mar-heter,mnar2-heter,mnar2-homog fedavg_ft 8
./scripts/run_scenario_gpu.sh hhip niid-t1 mcar,mar-heter,mar-homog,mnar2-heter,mnar2-homog fedavg_ft 8
./scripts/run_scenario_gpu.sh hhip niid-t2 mcar,mnar2-homog fedavg_ft 8