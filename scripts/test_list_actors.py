import httpx

def check_actor_status(client: httpx.Client):
    """Poll until actors are ready."""
    response = client.post("/get_ray_actors")
    status_data = response.json()
    print(f"Actor names: {status_data['actor_names']}")

def test_list_actors():
    client = httpx.Client(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=600.0
    )
    check_actor_status(client)

if __name__ == "__main__":
    test_list_actors()