
import os
import time
import json

from itertools import combinations, permutations


from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize, LoopException
from erdbeermet.visualize.BoxGraphVis import plot_box_graph

class Pipeline:
    def __init__(self, matrix_size, clocklike=False, circular=False):
        self.N = matrix_size
        self.clocklike = clocklike
        self.circular = circular
        self.branching_prob = 0.0
        self.scenario = None

    def recognition_original_algorithm(self, scenario, use_shortest_spike=False):

        if use_shortest_spike:
            try:
                start_time = time.time()
                recognition_tree = recognize(scenario.D, first_candidate_only=True, print_info=False, use_shortest_spike=True)
                timedelta = time.time() - start_time
            except(LoopException):
                return {"loop" : True}

        else:
            start_time = time.time()
            recognition_tree = recognize(scenario.D, first_candidate_only=True, print_info=False)
            timedelta = time.time() - start_time

        # Classify whether the distance matrix was correctly recognized as an R-Map.
        recognized_rmap = recognition_tree.successes > 0

        # Classify whether the final 4-leaf map after recognition matches the first 4 leaves of the simulation.
        cooptimal_solutions = 0
        for node in recognition_tree.preorder():
            if node.valid_ways == 1:
                final_4_leaf_map = node.V
                cooptimal_solutions += 1

        first_4_leaves_sim = [scenario.history[0][0],
                                scenario.history[0][2],
                                scenario.history[1][2],
                                scenario.history[2][2]]

        # leaves_match = first_4_leaves_sim in final_4_leaf_maps

        if recognized_rmap:
            leaves_match = first_4_leaves_sim == final_4_leaf_map


            # Measure divergence of the reconstructed steps from true steps of the simulation, e.g. by counting common triples.
            true_triples = [tuple(sorted(step[:3])) for step in scenario.history]

            recognition_triples = [node.R_step[:3] for node in recognition_tree.postorder() if node.R_step is not None and node.valid_ways > 0]

            overlap = len(set(recognition_triples).intersection(set(true_triples))) # / len(recognition_triples)
        else:
            leaves_match = False
            overlap = 0
            # Plot results
            result_dir = "results"
            #recognition_tree.visualize(save_as=os.path.join(result_dir, f'tree_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}.svg'))


        return {"runtime" : timedelta, 
                "recognized_rmap": recognized_rmap, 
                "leaves_match": leaves_match, 
                "overlap": overlap,
                "cooptimal_solutions": cooptimal_solutions, 
                "loop": False }

    def recognition_with_blocked_leaves(self, scenario, core_leaves_unknown=False, num_block_leaves=4):
        
        first_4_leaves_sim = [scenario.history[0][0],
                            scenario.history[0][2],
                            scenario.history[1][2],
                            scenario.history[2][2]]

        if core_leaves_unknown:
            for blocked_leaves_candidate in combinations(range(self.N), num_block_leaves):
                start_time = time.time()
                recognition_tree = recognize(scenario.D, first_candidate_only=True, print_info=False, blocked_leaves=blocked_leaves_candidate)
                timedelta = time.time() - start_time
                recognized_rmap = recognition_tree.successes > 0
                if recognized_rmap:
                    break

        else: 
            if num_block_leaves == 4:
                blocked_leaves = first_4_leaves_sim
            elif num_block_leaves == 3:
                blocked_leaves = [scenario.history[0][0],
                                    scenario.history[0][2],
                                    scenario.history[1][2]]
            else:
                print("blocked leaves must be 3 or 4")
                return

            start_time = time.time()
            recognition_tree = recognize(scenario.D, first_candidate_only=True, print_info=False, blocked_leaves=blocked_leaves)
            timedelta = time.time() - start_time

        # Classify whether the distance matrix was correctly recognized as an R-Map.
        recognized_rmap = recognition_tree.successes > 0

        # Classify whether the final 4-leaf map after recognition matches the first 4 leaves of the simulation.
        cooptimal_solutions = 0
        for node in recognition_tree.preorder():
            if node.valid_ways == 1:
                final_4_leaf_map = node.V
                cooptimal_solutions += 1

        # leaves_match = first_4_leaves_sim in final_4_leaf_maps

        if recognized_rmap:
            leaves_match = first_4_leaves_sim == final_4_leaf_map


            # Measure divergence of the reconstructed steps from true steps of the simulation, e.g. by counting common triples.
            true_triples = [tuple(sorted(step[:3])) for step in scenario.history]

            recognition_triples = [node.R_step[:3] for node in recognition_tree.postorder() if node.R_step is not None and node.valid_ways > 0]

            overlap = len(set(recognition_triples).intersection(set(true_triples))) # / len(recognition_triples)
        else:
            leaves_match = False
            overlap = 0

            # Plot results
            result_dir = "results"
            #recognition_tree.visualize(save_as=os.path.join(result_dir, f'tree_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}.svg'))


        return {"runtime" : timedelta, 
                "recognized_rmap": recognized_rmap, 
                "leaves_match": leaves_match, 
                "overlap": overlap,
                "cooptimal_solutions": cooptimal_solutions }

    def simulate(self):
        self.scenario = simulate(self.N, self.branching_prob, self.circular, self.clocklike)

    def run_original_algo(self):
        return self.recognition_original_algorithm(self.scenario)

    def run_blocked_leaves(self, core_leaves_unknown=False, num_blocked_leaves=4):
        return self.recognition_with_blocked_leaves(self.scenario, core_leaves_unknown=core_leaves_unknown, num_block_leaves=num_blocked_leaves)

    def run_shortest_spike(self):
        return self.recognition_original_algorithm(self.scenario, use_shortest_spike=True)

    def run_all(self):

        result_dir = "results"
        results = dict()



        results["original_algo"] = self.recognition_original_algorithm()

        if not results["original_algo"]["recognized_rmap"]:
            history_file = os.path.join(result_dir, f'history_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}')
            self.scenario.write_history(history_file)
            scenario_4only = load(history_file, stop_after=4)
            #plot_box_graph(scenario_4only.D, labels=['a', 'b', 'c', 'd'])

        results["shortest_spike"] = self.recognition_original_algorithm(self.scenario, use_shortest_spike=True)

        results["blocked_leaves"] = self.recognition_with_blocked_leaves(self.scenario, num_block_leaves=4)
        results["blocked_leaves_3"] = self.recognition_with_blocked_leaves(self.scenario, num_block_leaves=3)
        results["blocked_leaves_unknown_4"] = self.recognition_with_blocked_leaves(self.scenario, core_leaves_unknown=True, num_block_leaves=4)
        results["blocked_leaves_unknown_3"] = self.recognition_with_blocked_leaves(self.scenario, core_leaves_unknown=True, num_block_leaves=3)
        
        return results

if __name__ == "__main__":

    sample_size = 1000
    number_of_fails_original = 0
    number_of_fails_shortest_spike = 0
    number_of_fails_blocked_leaves = 0
    runtimes_blocked_leaves = []
    runtimes_original_algo = []
    runtimes_shortest_spike = []
    cooptimal_solutions_original_algo = 0
    cooptimal_solutions_blocked_leaves = 0
    cooptimal_solutions_shortest_spike = 0
    loop_count_shortest_spike = 0

    for i in range(sample_size):
        pipe = Pipeline(6)
        pipe.simulate()
        results_of_original_algo = pipe.run_original_algo()
        runtimes_original_algo.append(results_of_original_algo["runtime"])
        cooptimal_solutions_original_algo += results_of_original_algo["cooptimal_solutions"]
        if not results_of_original_algo["recognized_rmap"]:
            number_of_fails_original += 1
            results_of_shortest_spike = pipe.run_shortest_spike()
            if not results_of_shortest_spike["loop"]:
                runtimes_shortest_spike.append(results_of_shortest_spike["runtime"])
                cooptimal_solutions_shortest_spike += results_of_shortest_spike["cooptimal_solutions"]
                if not results_of_shortest_spike["recognized_rmap"]:
                    number_of_fails_shortest_spike += 1
            else:
                loop_count_shortest_spike += 1

            results_of_blocked_leaves = pipe.run_blocked_leaves()
            runtimes_blocked_leaves.append(results_of_blocked_leaves["runtime"])
            cooptimal_solutions_blocked_leaves += results_of_blocked_leaves["cooptimal_solutions"]
            if not results_of_blocked_leaves["recognized_rmap"]:
                number_of_fails_blocked_leaves += 1

    
    avg_runtime_original_algo = sum(runtimes_original_algo) / sample_size
    avg_runtime_blocked_leaves = sum(runtimes_blocked_leaves) / sample_size
    avg_runtime_shortest_spike = sum(runtimes_shortest_spike) / sample_size
    avg_cooptimal_solutions_original_algo = cooptimal_solutions_original_algo / sample_size
    avg_cooptimal_solutions_blocked_leaves = cooptimal_solutions_blocked_leaves / sample_size
    avg_cooptimal_solutions_shortest_spike = cooptimal_solutions_shortest_spike / sample_size

    results = {"original algo" : {"avg runtime" : avg_runtime_original_algo,
                        "failed recognitions" : number_of_fails_original,
                        "avg cooptimal solutions" : avg_cooptimal_solutions_original_algo},
                "blocked leaves" : {"avg runtime" : avg_runtime_blocked_leaves,
                        "failed recognitions" : number_of_fails_blocked_leaves,
                        "avg cooptimal solutions" : avg_cooptimal_solutions_blocked_leaves},
                "shortest spike" : {"avg runtime" : avg_runtime_shortest_spike,
                        "failed recognitions" : number_of_fails_shortest_spike,
                        "avg cooptimal solutions" : avg_cooptimal_solutions_shortest_spike,
                        "loops found" : loop_count_shortest_spike}
    }

    print(results)
    resultdir = "results"
    outfile = os.path.join(resultdir, f"results_samplesize{sample_size}.json")
    with open(outfile, "w") as f: 
        json.dump(results, f)

   
"""     results = dict()
    for matrix_size in [6]:
        results[f"N={matrix_size}"] = dict()
        for clocklike in [True, False]:
            results[f"N={matrix_size}"][f"clocklike={clocklike}"] = dict()
            for circular in [True, False]:
                runtimes_original_algo = []
                runtimes_blocked_leaves = []
                failed_recognitions_original_algo = 0
                failed_recognitions_blocked_leaves = 0
                cooptimal_solutions_original_algo = 0
                cooptimal_solutions_blocked_leaves = 0
                for i in range(sample_size):
                    pipe = Pipeline(matrix_size, clocklike=clocklike, circular=circular)
                    results_of_run = pipe.run()
                    runtimes_original_algo.append(results_of_run["original_algo"]["runtime"])
                    runtimes_blocked_leaves.append(results_of_run["blocked_leaves"]["runtime"])
                    if not results_of_run["original_algo"]["recognized_rmap"]:
                        failed_recognitions_original_algo += 1
                    if not results_of_run["blocked_leaves"]["recognized_rmap"]:
                        failed_recognitions_blocked_leaves += 1
                    cooptimal_solutions_original_algo += results_of_run["original_algo"]["cooptimal_solutions"]
                    cooptimal_solutions_blocked_leaves += results_of_run["blocked_leaves"]["cooptimal_solutions"]
                avg_runtime_original_algo = sum(runtimes_original_algo) / len(runtimes_original_algo)
                avg_runtime_blocked_leaves = sum(runtimes_blocked_leaves) / len(runtimes_blocked_leaves)
                avg_cooptimal_solutions_original_algo = cooptimal_solutions_original_algo / sample_size
                avg_cooptimal_solutions_blocked_leaves = cooptimal_solutions_blocked_leaves / sample_size
                results[f"N={matrix_size}"][f"clocklike={clocklike}"][f"circular={circular}"] = {"original algo" : {"avg runtime" : avg_runtime_original_algo,
                                                                                                                    "failed recognitions" : failed_recognitions_original_algo,
                                                                                                                    "avg cooptimal solutions" : avg_cooptimal_solutions_original_algo},
                                                                                                "blocked leaves" : {"avg runtime" : avg_runtime_blocked_leaves,
                                                                                                                    "failed recognitions" : failed_recognitions_blocked_leaves,
                                                                                                                    "avg cooptimal solutions" : avg_cooptimal_solutions_blocked_leaves}
                }

    print(results)
    resultdir = "results"
    outfile = os.path.join(resultdir, f"results_samplesize{sample_size}.json")
    with open(outfile, "w") as f: 
        json.dump(results, f) """


    