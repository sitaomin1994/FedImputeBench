from src.server import (
    Server,
    ServerICE,
    ServerJM,
)


def load_server(server: str, server_config: dict) -> Server:
    if server == 'ice':
        server = ServerICE()
    elif server == 'jm':
        server = ServerJM()
    else:
        raise ValueError(f"Server {server} is not supported")
    return server
