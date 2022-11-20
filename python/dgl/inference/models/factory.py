from .gat import DistGAT
from .gcn import DistGCN
from .sage import DistSAGE

def load_model(model_type, num_inputs, num_hiddens, num_outputs, num_layers, heads):
    if model_type == 'gcn':
        model = DistGCN(num_inputs, num_hiddens, num_outputs, num_layers)
    elif model_type == 'sage':
        model = DistSAGE(num_inputs, num_hiddens, num_outputs, num_layers)
    elif model_type == 'gat':
        heads = list(map(lambda x: int(x), heads.split(",")))
        model = DistGAT(num_inputs, num_hiddens, num_outputs, num_layers, heads)
    else:
        print(f"Unknown model_type: {model_type}")
        exit(-1)
    return model
