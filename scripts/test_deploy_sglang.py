from tinkerbell.server.inference.sglang import SGLangServer

server = SGLangServer(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
server.deploy_to_modal(gpu="H100:1")