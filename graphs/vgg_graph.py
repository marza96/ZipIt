from graphs.base_graph import BIGGraph, NodeType


class VGGGraph(BIGGraph):
    def __init__(self, model, cfg, bnorm=False):
        super().__init__(model)
        
        self.cfg   = cfg
        self.bnorm = bnorm

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)

        cnt = 0
        graph = []
        names = []

        for x in self.cfg:
            if x == "M":
                graph.append(f"layers.{cnt}") # MaxPool2d
                cnt += 1

                names.append("MaxPool2d")

            else:
                graph.append(f"layers.{cnt}") # Conv2d
                cnt += 1

                names.append("Conv2d")


                if self.bnorm is True:
                    graph.append(f"layers.{cnt}") # BatchNorm2d
                    cnt += 1

                    names.append("BatchNorm2d")


                graph.append(f"layers.{cnt}") # ReLU
                cnt += 1

                names.append("ReLU")
                
                graph.append(NodeType.PREFIX)

                names.append("PREF")

        graph.append(f"layers.{cnt}") # AvgPool2d
        cnt += 1

        names.append("AvgPool2d")

        graph.append(f"layers.{cnt}") # Linear
        cnt += 1

        names.append("classifier")

        if self.bnorm is True:
            graph.append(f"layers.{cnt}") # Batchnorm2d
            cnt += 1

            names.append("BatchNorm2d")

        graph.append(NodeType.OUTPUT)

        names.append("out")

        for idx, g in enumerate(graph):
            print(g, names[idx])
        
        self.add_nodes_from_sequence('', graph, input_node, sep='')

        return self


if __name__ == "__main__":
    from vgg import VGG

    model = VGG('VGG11', 1, num_classes=10).to("mps")

    graph = VGGGraph(
        model, 
        [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    ).graphify()

