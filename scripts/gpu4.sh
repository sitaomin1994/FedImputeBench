#!/bin/bash
# 5
./scripts/run_scenario_gpu.sh hhip iid-even mnar2-heter fedprox 8
./scripts/run_scenario_gpu.sh hhip iid-uneven mcar,mnar2-homog fedprox 8
./scripts/run_scenario_gpu.sh hhip niid-t1 mar-homog fedprox 8
./scripts/run_scenario_gpu.sh hhip niid-t2 mnar2-homog fedprox 8