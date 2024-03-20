from typing import Union, List

from src.workflows.workflow_ice import WorkflowICE
from src.workflows.workflow_icegrad import WorkflowICEGrad
from src.workflows.workflow_jm import WorkflowJM


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
    elif workflow_name == 'jm':
        return WorkflowJM(workflow_params)
    else:
        raise ValueError(f"Workflow {workflow_name} not supported")