
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
        self.result_dir = "results"


    def recognition(self, algorithm="original_algorithm", core_leaves_unknown=False, num_block_leaves=4):
        """
        algorithm: "original_algorithm" or "blocked_leaves", or "shortest_spike"
        """

        first_4_leaves_sim = [self.scenario.history[0][0],
                            self.scenario.history[0][2],
                            self.scenario.history[1][2],
                            self.scenario.history[2][2]]

        if algorithm == "blocked_leaves":

            if core_leaves_unknown:
                for blocked_leaves_candidate in combinations(range(self.N), num_block_leaves):
                    start_time = time.time()
                    recognition_tree = recognize(self.scenario.D, first_candidate_only=True, print_info=False, blocked_leaves=blocked_leaves_candidate)
                    timedelta = time.time() - start_time
                    recognized_rmap = recognition_tree.successes > 0
                    if recognized_rmap:
                        break

            else: 
                if num_block_leaves == 4:
                    blocked_leaves = first_4_leaves_sim
                elif num_block_leaves == 3:
                    blocked_leaves = [self.scenario.history[0][0],
                                        self.scenario.history[0][2],
                                        self.scenario.history[1][2]]
                else:
                    print("blocked leaves must be 3 or 4")
                    return

                start_time = time.time()
                recognition_tree = recognize(self.scenario.D, first_candidate_only=True, print_info=False, blocked_leaves=blocked_leaves)
                timedelta = time.time() - start_time

        elif algorithm == "original_algorithm":
            start_time = time.time()
            recognition_tree = recognize(self.scenario.D, first_candidate_only=True, print_info=False)
            timedelta = time.time() - start_time

        elif algorithm == "shortest_spike":
            try:
                start_time = time.time()
                recognition_tree = recognize(self.scenario.D, first_candidate_only=True, print_info=False, use_shortest_spike=True)
                timedelta = time.time() - start_time
            except(LoopException):
                return {"loop" : True}

        else:
            print("no algorithm with that name")
            return

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
            true_triples = [tuple(sorted(step[:3])) for step in self.scenario.history]

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
                "loop": False}

    def write_failed_recognition_output(self, fail_count=0, result_dir=None):      
        if not result_dir:
            result_dir = self.result_dir  
        result_dir = os.path.join(result_dir, "failed_scenarios") 
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)   
        history_file = os.path.join(result_dir, f'history_scenario_{fail_count}')
        self.scenario.write_history(history_file)
        #scenario_4only = load(history_file, stop_after=4)
        #plot_box_graph(scenario_4only.D, labels=['a', 'b', 'c', 'd'])

    
    def simulate(self):
        self.scenario = simulate(self.N, self.branching_prob, self.circular, self.clocklike)

    def run_original_algo(self):
        results = self.recognition(algorithm="original_algorithm")
        if not results["recognized_rmap"]:
            self.write_failed_recognition_output()
        return results

    def run_blocked_leaves(self, core_leaves_unknown=False, num_blocked_leaves=4):
        return self.recognition(algorithm="blocked_leaves", core_leaves_unknown=core_leaves_unknown, num_block_leaves=num_blocked_leaves)

    def run_shortest_spike(self):
        return self.recognition(algorithm="shortest_spike")

    def run_all(self):

        result_dir = "results"
        results = dict()

        for algorithm in ["original_algorithm", "blocked_leaves", "shortest_spike"]:
            
            if algorithm == "blocked_leaves":
                for num_blocked_leaves in [3,4]:
                    for core_leaves_unknown in [True, False]:
                        algo_results = self.recognition(algorithm=algorithm, core_leaves_unknown=core_leaves_unknown, num_block_leaves=num_blocked_leaves)
                        results[f"{algorithm}_{core_leaves_unknown}_{num_blocked_leaves}"] = algo_results
                        if not algo_results["recognized_rmap"]:
                            history_file = os.path.join(result_dir, f'history_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}')
                            self.scenario.write_history(history_file)
                            scenario_4only = load(history_file, stop_after=4)
                            plot_box_graph(scenario_4only.D, labels=['a', 'b', 'c', 'd'])
            else:
                algo_results = self.recognition(algorithm=algorithm)
                results[algorithm] = algo_results
                if not algo_results["recognized_rmap"]:
                    history_file = os.path.join(result_dir, f'history_{self.N}_{"circular" if self.circular else "noncircular"}_{"clocklike" if self.clocklike else "nonclocklike"}')
                    self.scenario.write_history(history_file)
                    scenario_4only = load(history_file, stop_after=4)
                    plot_box_graph(scenario_4only.D, labels=['a', 'b', 'c', 'd'])

        return results
                

if __name__ == "__main__":

    ### Adapt your parameters here #########
    sample_size = 20000
    num_leaves = 7
    clocklike = False
    circular = True

    algorithm = "shortest_spike"
    base_resultdir = "results_wp4"

    ########################################

    if circular:
        mode = "circular"
    elif clocklike:
        mode = "clocklike"
    else:
        mode = "normal"

    resultdir = os.path.join(base_resultdir, f"n{num_leaves}", mode)
    if not os.path.isdir(resultdir):
        os.makedirs(resultdir)

    cooptimal_solutions = 0
    runtimes = 0
    number_of_fails = 0

    for i in range(sample_size):
        if i%100 == 0:
            print(f"Simulated {i} samples")
        pipe = Pipeline(num_leaves,clocklike=clocklike, circular=circular)
        pipe.simulate()
        results = pipe.recognition(algorithm=algorithm, core_leaves_unknown=True)
        if "recognized_rmap" in results.keys():
            if not results["recognized_rmap"]:
                number_of_fails += 1
                pipe.write_failed_recognition_output(fail_count=number_of_fails, result_dir=resultdir)
            runtimes += results["runtime"]
            cooptimal_solutions += results["cooptimal_solutions"]
        elif "loop" in results.keys():
            if results["loop"]:
                number_of_fails += 1
        else:
            print("Something went wrong")
    avg_runtime = runtimes / sample_size
    avg_cooptimal_solutions = cooptimal_solutions / sample_size

    results = {"avg runtime" : avg_runtime,
                "failed recognitions" : number_of_fails,
                "avg cooptimal solutions" : avg_cooptimal_solutions}

    print(results)
    outfile = os.path.join(resultdir, f"results_samplesize{sample_size}_{num_leaves}leaves_{mode}.json")
    with open(outfile, "w") as f: 
        json.dump(results, f)



""" 
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
        if i%100 == 0:
            print(f"Simulated {i} samples")
        pipe = Pipeline(num_leaves,clocklike=clocklike, circular=circular)
        pipe.simulate()
        results_of_original_algo = pipe.run_original_algo()
        pipe.run_shortest_spike()
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
    outfile = os.path.join(resultdir, f"results_samplesize{sample_size}_{num_leaves}leaves.json")
    with open(outfile, "w") as f: 
        json.dump(results, f) """

   



    