#!/bin/bash
./scripts/run_scenario_cpu2.sh codrna iid-even,iid-uneven,niid-f1,niid-f2
./scripts/run_scenario_cpu2.sh california iid-even,iid-uneven,niid-t1,niid-t2
./scripts/run_scenario_cpu2.sh dvisits iid-even,iid-uneven,niid-t1,niid-t2
./scripts/run_scenario_cpu2.sh hhip iid-even,iid-uneven,niid-t1,niid-t2
./scripts/run_scenario_cpu2.sh vehicle iid-even,iid-uneven,niid-f1,niid-f2
./scripts/run_scenario_cpu2.sh school_pca iid-even,iid-uneven,niid-f1,niid-f2

