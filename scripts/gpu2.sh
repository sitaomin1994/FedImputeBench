#!/bin/bash
# 9
./scripts/run_scenario_gpu.sh hhip iid-even mar-homog miwae fedavg 8
./scripts/run_scenario_gpu.sh hhip iid-uneven mcar,mnar2-heter,mnar2-homog miwae fedavg 8
./scripts/run_scenario_gpu.sh hhip niid-t2 mar-heter,mnar2-homog miwae fedavg 8