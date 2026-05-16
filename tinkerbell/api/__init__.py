def deploy_on_modal(*args, **kwargs):
    from tinkerbell.api.deploy import deploy_on_modal as _deploy_on_modal

    return _deploy_on_modal(*args, **kwargs)


def deploy_service(*args, **kwargs):
    from tinkerbell.api.server import deploy_service as _deploy_service

    return _deploy_service(*args, **kwargs)


def __getattr__(name: str):
    if name in {"APP", "TinkerbellServer"}:
        from tinkerbell.api import server

        return getattr(server, name)
    raise AttributeError(name)


__all__ = ["APP", "TinkerbellServer", "deploy_on_modal", "deploy_service"]
