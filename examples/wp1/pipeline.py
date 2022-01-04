
import os


from erdbeermet.simulation import simulate
from erdbeermet.recognition import recognize

class Pipeline:
    def __init__(self, matrix_size, clocklike=False, circular=False):
        self.N = matrix_size
        self.clocklike = clocklike
        self.circular = circular
        self.branching_prob = 0.0

    def run(self):

        scenario = simulate(self.N, self.branching_prob, self.circular, self.clocklike)

        recognition_tree = recognize(scenario.D, print_info=False)


        # Classify whether the distance matrix was correctly recognized as an R-Map.
        recognized_rmap = recognition_tree.successes > 0

        # Classify whether the final 4-leaf map after recognition matches the first 4 leaves of the simulation.
        final_4_leaf_maps = []
        for node in recognition_tree.preorder():
            if node.valid_ways == 1:
                    final_4_leaf_maps.append(node.V)

        first_4_leaves_sim = [scenario.history[0][0],
                                scenario.history[0][2],
                                scenario.history[1][2],
                                scenario.history[2][2]]

        leaves_match = first_4_leaves_sim in final_4_leaf_maps


        # Measure divergence of the reconstructed steps from true steps of the simulation, e.g. by counting common triples.
        true_triples = [tuple(sorted(step[:3])) for step in scenario.history]
        print(true_triples)

        recognition_triples = [node.R_step[:3] for node in recognition_tree.postorder() if node.R_step is not None and node.valid_ways > 0]
        print(recognition_triples)
        overlap = len(set(recognition_triples).intersection(set(true_triples))) / len(recognition_triples)
        print(overlap)

if __name__ == "__main__":
    for matrix_size in [6]:
        pipe = Pipeline(matrix_size)
        pipe.run()