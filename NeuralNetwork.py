import random
from math import exp

#  Works for any NN with 1 hidden-layer
class NeuralNetwork:
    node_id_counter = 0

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes, bias_one, bias_two, learning_rate):
        self.learning_rate = learning_rate
        self.input_nodes = self.create_nodes(num_input_nodes)
        self.hidden_nodes = self.create_nodes(num_hidden_nodes)
        self.output_nodes = self.create_nodes(num_output_nodes)
        self.weights = self.create_weights(num_input_nodes, num_hidden_nodes, num_output_nodes)
        self.biases = self.create_biases(bias_one, bias_two)
        print(f'Neural network created: {num_input_nodes} {num_hidden_nodes} {num_output_nodes}')

        self.randomize_weights()

        # override weights
        #  self.weights[0].out = 0.15
        #  self.weights[1].out = 0.25
        #  self.weights[2].out = 0.20
        #  self.weights[3].out = 0.30
        #  self.weights[4].out = 0.40
        #  self.weights[5].out = 0.50
        #  self.weights[6].out = 0.45
        #  self.weights[7].out = 0.55

    class Node:
        def __init__(self, neural_network_ref):
            self.id = neural_network_ref.node_id_counter
            self.net = 0  # only for hidden and output-layer
            self.out = 0
            self.target = 0  # only for output-layer
            neural_network_ref.node_id_counter += 1

    class Weight:
        def __init__(self, val, node_one_id, node_two_id):
            self.out = val
            self.nodes = [node_one_id, node_two_id]

    def create_nodes(self, num_nodes):
        return [self.Node(self) for _ in range(num_nodes)]

    def create_weights(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        weights = []
        for x in range(num_input_nodes):
            node_one_id = self.input_nodes[x].id
            for y in range(num_hidden_nodes):
                node_two_id = self.hidden_nodes[y].id
                weights.append(self.Weight(0, node_one_id, node_two_id))
        for x in range(num_hidden_nodes):
            node_one_id = self.hidden_nodes[x].id
            for y in range(num_output_nodes):
                node_two_id = self.output_nodes[y].id
                weights.append(self.Weight(0, node_one_id, node_two_id))
        return weights

    def create_biases(self, bias_one, bias_two):
        return [bias_one, bias_two]

    def forward_pass(self, input_node_vals, bias_one, bias_two):

        # reset nodes and bias
        self.reset()

        # set inputs
        self.biases = [bias_one, bias_two]
        for index, x in enumerate(input_node_vals):
            self.input_nodes[index].out = x

        # hidden-layer net and out calculations
        for x in self.hidden_nodes:
            for y in self.weights:
                if y.nodes[1] == x.id:
                    x.net += (y.out * next((_ for _ in self.input_nodes if _.id == y.nodes[0]), None).out)
            x.net += self.biases[0]
            x.out = 1 / (1 + exp(-x.net))

        # output-layer net and out calculations
        for x in self.output_nodes:
            for y in self.weights:
                if y.nodes[1] == x.id:
                    x.net += (y.out * next((_ for _ in self.hidden_nodes if _.id == y.nodes[0]), None).out)
            x.net += self.biases[1]
            x.out = 1 / (1 + exp(-x.net))

    def back_propagate(self, output_node_vals):
        print(f'Back propagating...')
        cost = self.get_network_cost(output_node_vals)
        weight_deltas = [self.Weight(0, x.nodes[0], x.nodes[1]) for x in self.weights]
        new_weights = [self.Weight(x.out, x.nodes[0], x.nodes[1]) for x in self.weights]

        # set targets
        for index in range(len(self.output_nodes)):
            self.output_nodes[index].target = output_node_vals[index]

        # calculate weight deltas
        for index, x in enumerate(self.weights):
            start_node = None
            end_node = None
            for y in self.input_nodes:
                if y.id == x.nodes[0]:
                    start_node = y
            for y in self.hidden_nodes:
                if y.id == x.nodes[0]:
                    start_node = y
                if y.id == x.nodes[1]:
                    end_node = y
            for y in self.output_nodes:
                if y.id == x.nodes[1]:
                    end_node = y
            if end_node in self.output_nodes:
                # calculate output-layer weight deltas
                weight_deltas[index].out = -(start_node.target - end_node.out) * end_node.out * (1 - end_node.out) * start_node.out
            else:
                # calculate hidden-layer weight deltas
                sum = 0
                for y in self.output_nodes:
                    hidden_to_output_weight_val = 0
                    for z in self.weights:
                        if z.nodes[0] == x.nodes[1] and z.nodes[1] == y.id:
                            hidden_to_output_weight_val = z.out
                    sum += (-y.target + y.out) * (y.out * (1 - y.out)) * hidden_to_output_weight_val
                weight_deltas[index].out = sum * (end_node.out * (1 - end_node.out)) * start_node.out

        # set new weights
        for index in range(len(weight_deltas)):
            self.weights[index].out = self.weights[index].out - (self.learning_rate * weight_deltas[index].out)

    def get_network_cost(self, output_node_vals):
        cost = 0
        for index, x in enumerate(self.output_nodes):
            output_cost = (1 / 2) * (output_node_vals[index] - x.out) ** 2
            cost += output_cost
        return cost

    def train(self, input_node_vals, output_node_vals, bias_one, bias_two):

        classification_index = 0

        # guess
        self.forward_pass(input_node_vals, bias_one, bias_two)

        # determine classification
        for index, x in enumerate(self.output_nodes):
            if (classification_index == -1) or (self.output_nodes[classification_index].out < x.out):
                classification_index = index

        # learn
        self.back_propagate(output_node_vals)

        return classification_index

    def reset(self):
        self.biases = [0, 0]
        for x in self.input_nodes:
            x.net = x.out = 0
        for x in self.hidden_nodes:
            x.net = x.out = 0
        for x in self.output_nodes:
            x.net = x.out = 0

    def randomize_weights(self):
        for x in self.weights:
            x.out = float('%.2f' % random.random())

    def inspect(self):
        print(f'~~~  STATUS ~~~')
        print(f'INPUT NODES: {[_.out for _ in self.input_nodes]}')
        print(f'HIDDEN NODES: {["(" + str(_.net) + ", " + str(_.out) + ")" for _ in self.hidden_nodes]}')
        print(f'OUTPUT NODES: {["(" + str(_.net) + ", " + str(_.out) + ")" for _ in self.output_nodes]}')
        print(f'BIASES: {[_ for _ in self.biases]}')
        print(f'WEIGHTS: {["(" + str(_.out) + ", " + str(_.nodes) + ")" for _ in self.weights]}')
        print(f'~~~~~~~~~~~~~~~')
