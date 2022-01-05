
import os
import time


from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize
from erdbeermet.visualize.BoxGraphVis import plot_box_graph

class Pipeline:
    def __init__(self, matrix_size, clocklike=False, circular=False):
        self.N = matrix_size
        self.clocklike = clocklike
        self.circular = circular
        self.branching_prob = 0.0

    def recognition_original_algorithm(self, scenario):

        start_time = time.time()
        recognition_tree = recognize(scenario.D, first_candidate_only=True, print_info=False)
        timedelta = time.time() - start_time

        # Classify whether the distance matrix was correctly recognized as an R-Map.
        recognized_rmap = recognition_tree.successes > 0

        # Classify whether the final 4-leaf map after recognition matches the first 4 leaves of the simulation.
        final_4_leaf_maps = []
        for node in recognition_tree.preorder():
            if node.valid_ways == 1:
                final_4_leaf_map = node.V
                break
                # final_4_leaf_maps.append(node.V)

        first_4_leaves_sim = [scenario.history[0][0],
                                scenario.history[0][2],
                                scenario.history[1][2],
                                scenario.history[2][2]]

        # leaves_match = first_4_leaves_sim in final_4_leaf_maps
        leaves_match = first_4_leaves_sim == final_4_leaf_map


        # Measure divergence of the reconstructed steps from true steps of the simulation, e.g. by counting common triples.
        true_triples = [tuple(sorted(step[:3])) for step in scenario.history]

        recognition_triples = [node.R_step[:3] for node in recognition_tree.postorder() if node.R_step is not None and node.valid_ways > 0]

        overlap = len(set(recognition_triples).intersection(set(true_triples))) # / len(recognition_triples)

        # Plot results
        result_dir = "results"
        #recognition_tree.visualize(save_as=os.path.join(result_dir, f'history_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}.svg'))


        return timedelta, recognized_rmap, leaves_match, overlap

    def recognize_with_blocked_leaves(self, scenario):
        # WP3
        pass

    def run(self):

        result_dir = "results"

        scenario = simulate(self.N, self.branching_prob, self.circular, self.clocklike)

        timedelta1, recognized_rmap1, leavesmatch1, overlap1 = self.recognition_original_algorithm(scenario)

        if not recognized_rmap1:
            history_file = os.path.join(result_dir, f'history_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}')
            scenario.write_history(history_file)
            scenario_4only = load(history_file, stop_after=4)
            plot_box_graph(scenario_4only.D, labels=['a', 'b', 'c', 'd'])

        return timedelta1, recognized_rmap1, leavesmatch1, overlap1

if __name__ == "__main__":

    for matrix_size in [8]:
        runtimes = []
        failed_recognitions = 0
        for i in range(10):
            pipe1 = Pipeline(matrix_size)
            runtime1, recognized_rmap1, leavesmatch1, overlap1 = pipe1.run()
            runtimes.append(runtime1)
            if not recognized_rmap1:
                failed_recognitions += 1
        avg_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime: {avg_runtime}")
        print(f"Failed recognitions: {failed_recognitions}")

    for matrix_size in [8]:
        runtimes = []
        failed_recognitions = 0
        for i in range(10):
            pipe1 = Pipeline(matrix_size, clocklike=True)
            runtime1, recognized_rmap1, leavesmatch1, overlap1 = pipe1.run()
            runtimes.append(runtime1)
            if not recognized_rmap1:
                failed_recognitions += 1
        avg_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime: {avg_runtime}")
        print(f"Failed recognitions: {failed_recognitions}")


    for matrix_size in [8]:
        runtimes = []
        failed_recognitions = 0
        for i in range(10):
            pipe1 = Pipeline(matrix_size, circular=True)
            runtime1, recognized_rmap1, leavesmatch1, overlap1 = pipe1.run()
            runtimes.append(runtime1)
            if not recognized_rmap1:
                failed_recognitions += 1
        avg_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime: {avg_runtime}")
        print(f"Failed recognitions: {failed_recognitions}")
            