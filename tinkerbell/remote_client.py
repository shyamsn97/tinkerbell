import abc


class RemoteClient(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def deploy(self, config: dict) -> None:
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        pass

    @abc.abstractmethod
    def health(self) -> None:
        pass
