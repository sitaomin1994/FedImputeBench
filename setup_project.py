import os
import json

# Parse the JSON structure
with open("settings.json") as f:
    project_setting = json.load(f)

########################################################################################################################
# Create directories based on the JSON configuration
data_dir = project_setting['data_dir'] if 'data_dir' in project_setting else 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
print(f"Setting up data directory in {'./' + data_dir}")

########################################################################################################################
# configuration directory
config_dir = project_setting['exp_config_dir'] if 'exp_config_dir' in project_setting else 'config'
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
print(f"Setting up configuration directory in {'./' + config_dir}")

########################################################################################################################
# If it's the experiment result directory, also create the subdirectories
if "results" not in project_setting:
    results_base = "results"
    processed_result_dir = "processed_results"
    raw_result_dir = "raw_results"
else:
    results_base = project_setting['results']['base_dir'] if 'base_dir' in project_setting['results'] else 'results'
    processed_result_dir = project_setting['results']['processed_dir'] if 'processed_dir' in project_setting[
        'results'] else 'processed_results'
    raw_result_dir = project_setting['results']['raw_dir'] if 'raw_dir' in project_setting['results'] else 'raw_results'

if not os.path.exists(results_base):
    os.makedirs(results_base)
    os.makedirs(os.path.join(results_base, processed_result_dir))
    os.makedirs(os.path.join(results_base, raw_result_dir))
print(f"Setting up results directory in {'./' + results_base}")

########################################################################################################################
notebook_dir = project_setting['notebook_dir'] if 'notebook_dir' in project_setting else 'notebooks'
if not os.path.exists(notebook_dir):
    os.makedirs(notebook_dir)
print(f"Setting up notebook directory in {'./' + notebook_dir}")

########################################################################################################################
script_dir = project_setting['script_dir'] if 'script_dir' in project_setting else 'scripts'
if not os.path.exists(script_dir):
    os.makedirs(script_dir)
print(f"Setting up script directory in {'./' + script_dir}")

########################################################################################################################
script_dir = project_setting['scenario_dir'] if 'scenario_dir' in project_setting else 'scenario_data'
if not os.path.exists(script_dir):
    os.makedirs(script_dir)
print(f"Setting up script directory in {'./' + script_dir}")

########################################################################################################################
# create test_directory
test_dir = 'tests'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
print(f"Setting up test directory in {'./' + test_dir}")

########################################################################################################################
# create src directory
src_dir = 'FedImpute'
if not os.path.exists(src_dir):
    os.makedirs(src_dir)
    print(f"Setting up src directory in {'./' + src_dir}")
    # loaders
    loader_dir = os.path.join('./' + src_dir + '/loaders/')
    if not os.path.exists(loader_dir):
        os.makedirs(loader_dir)
        print(f"Setting up loader directory in {loader_dir}")
    with open(os.path.join(loader_dir, '__init__.py'), 'w') as f:
        f.write('')

    # modules
    modules_dir = os.path.join('./' + src_dir + '/modules/')
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir)
        print(f"Setting up modules directory in {modules_dir}")
    with open(os.path.join(modules_dir, '__init__.py'), 'w') as f:
        f.write('')


# Inform the user that the directories have been created
print("Project directories have been set up successfully.")
