#!/bin/bash
# 7
./scripts/run_scenario_gpu.sh hhip iid-even mcar,mar-homog local 8
./scripts/run_scenario_gpu.sh hhip iid-uneven mnar2-heter local 8
./scripts/run_scenario_gpu.sh hhip niid-t1 mcar,mnar2-heter local 8
./scripts/run_scenario_gpu.sh hhip niid-t2 mar-heter,mnar2-homog local 8