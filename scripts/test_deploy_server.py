from tinkerbell.server.training.torch_server import TorchServer

server = TorchServer()

if __name__ == "__main__":
    server.deploy_to_modal()