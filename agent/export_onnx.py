import torch
import onnx
from agent import ReversiNet

device = torch.device("cpu")
model = ReversiNet()
model_path = os.path.join(os.path.dirname(__file__), "reversi_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

dummy_input = torch.randn(1, 2, 8, 8)

# use legacy torchscript exporter to keep all weights embedded
torch.onnx.export(
    model,
    (dummy_input,),
    os.path.join(os.path.dirname(__file__), "../web/reversi_model.onnx"),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["policy", "value"],
    dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}, "value": {0: "batch_size"}},
    dynamo=False
)

# verify no external data references remain
web_onnx_path = os.path.join(os.path.dirname(__file__), "../web/reversi_model.onnx")
model_onnx = onnx.load(web_onnx_path, load_external_data=False)
for t in model_onnx.graph.initializer:
    if t.external_data:
        print(f"WARNING: {t.name} has external data reference, fixing...")
        # load the external data and embed it
        model_onnx = onnx.load(web_onnx_path, load_external_data=True)
        onnx.save(model_onnx, web_onnx_path)
        break

print("exported to web/reversi_model.onnx")
