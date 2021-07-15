from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    neural_network = NeuralNetwork(2, 2, 2, 0, 0, 0.5)
    classification = neural_network.train([0.05, 0.10], [0.01, 0.99], 0.35, 0.60)
    print(f'Classification index: {classification}')
    neural_network.inspect()
