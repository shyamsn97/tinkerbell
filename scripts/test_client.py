import httpx

if __name__ == "__main__":
    client = httpx.Client(base_url="https://jesterlabs--training-service.modal.run", timeout=300.0)
    response = client.get("/health")
    print(response.json())