from tinkerbell.server.inference.sglang import SGLangServer

server = SGLangServer(model_name="Qwen/Qwen3-0.6B")
server.deploy_to_modal(gpu="any:1")