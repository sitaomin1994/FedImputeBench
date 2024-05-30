#!/bin/bash
# 5
./scripts/run_scenario_gpu.sh hhip iid-even mnar2-heter miwae fedprox 8
./scripts/run_scenario_gpu.sh hhip iid-uneven mcar,mnar2-homog miwae fedprox 8
./scripts/run_scenario_gpu.sh hhip niid-t1 mar-homog miwae fedprox 8
./scripts/run_scenario_gpu.sh hhip niid-t2 mnar2-homog miwae fedprox 8