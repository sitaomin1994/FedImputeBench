#!/bin/bash
# 17
./scripts/run_scenario_gpu.sh hhip iid-even mcar,mnar2-heter,mnar2-homog central 1
./scripts/run_scenario_gpu.sh hhip iid-uneven mar-heter,mar-homog,mnar2-heter,mnar2-homog central 1
./scripts/run_scenario_gpu.sh hhip niid-t1,niid-t2 mcar,mar-homog,mar-heter,mnar2-heter,mnar2-homog central 1