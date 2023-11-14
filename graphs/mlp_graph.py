from .base_graph import BIGGraph, NodeType

class MLPGraph(BIGGraph):
    def __init__(self, model, num_layers):
        super().__init__(model)

        self.num_layers = num_layers

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)

        graph = []

        graph.append('fc1')
        graph.append(NodeType.PREFIX)

        for i in range(self.num_layers * 2):
            graph.append('layers.' + str(i))
            if i % 2 != 0:
                graph.append(NodeType.PREFIX)

        graph.append('fc2')
        graph.append(NodeType.OUTPUT)

        self.add_nodes_from_sequence('', graph, input_node, sep='')

        return self
    
