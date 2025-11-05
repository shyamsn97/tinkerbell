from pydantic import BaseModel


class RemoteFuture(BaseModel):
    request_id: str

