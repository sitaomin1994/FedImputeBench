from src.client import Client
from src.server import Server
from typing import List
from typing import Union, List

from src.workflows.workflow_ice import WorkflowICE
from src.workflows.workflow_icegrad import WorkflowICEGrad
from src.workflows.workflow_jm import WorkflowJM


def setup_clients(
        clients_data, clients_seeds, data_config: dict,
        imp_model_name: str, imp_model_params: dict, fed_strategy: str, fed_strategy_client_params: dict,
) -> List[Client]:
    clients = []
    for client_id, (client_data, client_seed) in enumerate(zip(clients_data, clients_seeds)):
        client = Client(
            client_id, train_data=client_data[0], test_data=client_data[1], X_train_ms=client_data[2],
            data_config=data_config, imp_model_name=imp_model_name, imp_model_params=imp_model_params,
            fed_strategy=fed_strategy, fed_strategy_params=fed_strategy_client_params, seed=client_seed,
            client_config={}
        )
        clients.append(client)

    return clients


def setup_server(
        fed_strategy: str, fed_strategy_params: dict, server_config: dict
) -> Server:
    server = Server(
        fed_strategy, fed_strategy_params, server_config
    )

    return server


def load_workflow(
        workflow_name: str,
        workflow_params: dict,
) -> Union[WorkflowICE, WorkflowICEGrad, WorkflowJM]:
    """
    Load the workflow based on the workflow name
    """
    if workflow_name == 'ice':
        return WorkflowICE(workflow_params)
    elif workflow_name == 'icegrad':
        return WorkflowICEGrad(workflow_params)
    elif workflow_name == 'vae':
        return WorkflowJM(workflow_params)
    else:
        raise ValueError(f"Workflow {workflow_name} not supported")
