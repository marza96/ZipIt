from graphs.base_graph import BIGGraph, NodeType


class MLPGraph(BIGGraph):
    def __init__(self, model, num_layers, bnorm=False):
        super().__init__(model)

        self.bnorm      = bnorm
        self.num_layers = num_layers

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)

        cnt   = 0
        graph = []

        for i in range(-1, self.num_layers):
            graph.append(f"layers.{cnt}") # nn.Linear
            cnt += 1

            if self.bnorm is True:
                graph.append(f"layers.{cnt}") # nn.Batchnorm1d
                cnt += 1

            if i == self.num_layers - 1:
                break

            graph.append(f"layers.{cnt}") # nn.ReLU
            cnt += 1

            graph.append(NodeType.PREFIX)

        graph.append(NodeType.OUTPUT)
        
        self.add_nodes_from_sequence('', graph, input_node, sep='')

        return self
    