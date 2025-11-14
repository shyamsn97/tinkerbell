from tinkerbell.client.service import ServiceClient

if __name__ == "__main__":
    client = ServiceClient(base_url="https://jesterlabs--training-service.modal.run", timeout=300.0)
    print(client.get_health())