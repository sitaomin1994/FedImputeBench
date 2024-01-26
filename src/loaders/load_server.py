from src.server.server_fedavg import FedAvgServer
from src.server.server_local import LocalServer


def load_server(server, clients, server_config):
    if server == 'fedavg':
        server = FedAvgServer(clients, server_config)
    elif server == 'fedavg-enc':
        server = FedAvgServer(clients, server_config, layer_keys=['encoder'])
    elif server == 'fedavg-dec':
        server = FedAvgServer(clients, server_config, layer_keys=['decoder'])
    elif server == 'fedavg-enc-dec':
        server = FedAvgServer(clients, server_config, layer_keys=['encoder', 'decoder'])
    elif server == 'fedavg-masknet':
        server = FedAvgServer(clients, server_config, layer_keys=['mask'])
    elif server == 'fedavg-enc-masknet':
        server = FedAvgServer(clients, server_config, layer_keys=['encoder', 'mask'])
    elif server == 'fedavg-dec-masknet':
        server = FedAvgServer(clients, server_config, layer_keys=['decoder', 'mask'])
    elif server == 'local':
        server = LocalServer(clients, server_config)
    else:
        raise ValueError(f"Server {server} is not supported")
    return server
