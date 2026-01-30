import os
import random
import re
import time
from pdb import run

import ale_bench

from kapso.environment.handlers.base import ProblemHandler, ProblemRunResult
from kapso.core import llm as llm_utils

LITE_VERSION_PROBLEMS_LIST = ["ahc008", "ahc011", "ahc015", "ahc016", "ahc024", "ahc025", "ahc026", "ahc027", "ahc039", "ahc046"]
LITE_PROBLEMS_INFO = {
    'ahc002': {'contestants': 824, 'maximize_scoring': True}, 
    'ahc005': {'contestants': 824, 'maximize_scoring': True}, 
    'ahc006': {'contestants': 824, 'maximize_scoring': True}, 
    'ahc008': {'contestants': 824, 'maximize_scoring': True}, 
    'ahc011': {'contestants': 926, 'maximize_scoring': True},
    'ahc015': {'contestants': 779, 'maximize_scoring': True}, 
    'ahc016': {'contestants': 1047, 'maximize_scoring': True}, 
    'ahc024': {'contestants': 664, 'maximize_scoring': True}, 
    'ahc025': {'contestants': 879, 'maximize_scoring': False}, 
    'ahc026': {'contestants': 740, 'maximize_scoring': True}, 
    'ahc027': {'contestants': 999, 'maximize_scoring': False}, 
    'ahc039': {'contestants': 683, 'maximize_scoring': True}, 
    'ahc046': {'contestants': 939, 'maximize_scoring': True},
}

RUN_COUNTS = 4
MAXIMUM_CONCURENT_RUNS = 7

class AleBench(ProblemHandler):
    def __init__(self, problem_id = "ahc008"):
        # Initialize with domain knowledge as additional context
        super().__init__(additional_context=self._get_domain_knowledge())
        self.session = ale_bench.start(
            problem_id=problem_id,
            lite_version=False,
            num_workers=25,
            run_visualization_server=False,
        )
        self.currently_running = 0
        self.llm = llm_utils.LLMBackend()
        self.problem_id = problem_id
        self.maximize_scoring = LITE_PROBLEMS_INFO[problem_id]['maximize_scoring']
        self._set_problem_context(problem_id)
        

    def run(self, file_path, run_data_dir, solution="", code_language="cpp23", debug=False, feedback=False):
        while self.currently_running >= MAXIMUM_CONCURENT_RUNS:
            time.sleep(1)
        self.currently_running += 1
        code = self._prepare_code_from_file(file_path + "/main.cpp")
        run_result = self.session.public_eval(code, code_language=code_language)
        run_had_error = (
            (run_result.overall_judge_result != "ACCEPTED")
            or max([case.absolute_score == 0 or case.judge_result != "ACCEPTED" for case in run_result.case_results])
        )
        score = run_result.overall_absolute_score
        buggy_case = ""
        error_message = ""
        detailed_output = ""
        output = ""
        feedbacks = ""
        if len(run_result.case_results) and run_had_error:
            buggy_cases = list(filter(lambda x: x.absolute_score == 0 or x.judge_result != "ACCEPTED", run_result.case_results))
            buggy_case = str(buggy_cases[-1])
            error_message = str(buggy_cases[-1].message)
            if len(buggy_case) > 2000:
                buggy_case = buggy_case[:1000] + "..." + buggy_case[-1000:]
            if len(error_message) > 2000:
                error_message = error_message[:1000] + "..." + error_message[-1000:]
        if len(run_result.case_results) and not run_had_error:        
            detailed_output = str(random.sample(run_result.case_results, 5))
            max_run_time = max(run_result.case_results, key=lambda x: float(x.execution_time)).execution_time
            output = "Code execution time in experimentation: " + str(max_run_time) + " seconds"
            score = run_result.overall_absolute_score / RUN_COUNTS
            for i in range(1, RUN_COUNTS):
                score += self.session.public_eval(code, code_language=code_language).overall_absolute_score / RUN_COUNTS
            if feedback:
                feedbacks = self.get_feedback_on_tests(solution, code, random.sample(run_result.case_results, 5))
        self.currently_running -= 1
        return ProblemRunResult(
            detailed_output=detailed_output,
            score=score,
            output=output,
            run_had_error=run_had_error,
            error_message=run_result.overall_judge_result + "\n" + error_message,
            error_details=run_result.overall_judge_result + "\n" + buggy_case,
            feedbacks=feedbacks
        )
    
    def final_evaluate(self, file_path, code_language="cpp23"):
        code = self._prepare_code_from_file(file_path + "/main.cpp")
        private_result, final_rank, final_performance = self.session.private_eval(
            code, code_language=code_language
        )
        return {
            "final_rank": final_rank,
            "rank_percentage": round((final_rank / LITE_PROBLEMS_INFO[self.problem_id]['contestants']) * 100, 2),
            "final_performance": final_performance,
            "private_result": private_result.overall_absolute_score,
        }
 
    def get_problem_context(self, budget_progress = 0):
        return self.problem_context
        
    def get_lite_problems_list(self):
        return LITE_VERSION_PROBLEMS_LIST

    def _set_problem_context(self, problem_id):
        cache_file = f"tmp/ale/{problem_id}/context.txt"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f: 
                self.problem_context = f.read()
        else:
            self.problem_context = self._generate_problem_context()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f: 
                f.write(self.problem_context)
        self.problem_context += f""" \n\n
            You are a grandmaster algorithm designer and programmer and your goal to get the highest rank in the atcoder contest.
            # Problem requirements:
            - It is critical to completely understand the problem statement and its requirements to avoid getting WA (wrong answer) and TLE (time limit exceeded).
            - Make sure to always consider highest value of inputs to avoid time limit for your code execution. The cpp23 can perform only up to 1e8 simple operations per second, so perform no more operations than (time limit seconds * 1e8).
            - Score 0 means invalid solution.
            - In this problem the {"higher" if self.maximize_scoring else "lower"} score is better.
            - It is of utmost importance for your solution to be aware of requirements and constraints of the problem in the solution to avoid getting WA (wrong answer). 

            # Tips:
                - Always consider a proportion of time for input and output operations to avoid TIME LIMIT. So always consider io time plus 100 ms plus code runtime time to avoid time limit.
                    -- For example if the time limit is 2 seconds, you must only use 1.9 seconds if it.
                - Always calculate the time complexity of each step in your solution and idea generation, and the total runtime must be exact and in seconds based on the number of operations. For example do not round down 400 operations to O(100) in your calculations.
                - Althought you must avoid time limit but make sure to use the limited time as much as possible to get the highest score.
                - If the solution has more than one step, make sure to distribute the time limit among them efficiently.
                - consider adding compiler optimization pragmas or directives if it helps your code to run faster.
                - Make sure to use the fastest operations and data structures for every part of the code.
                - It is ok to put small pieces of codes in the solution generation for the better understanding of coder when helpful or necessary.
                
            # Solution requirements:
            - If run time allows always add more algorithms to the current best performing solutions.
            - Avoid inefficient algorithms like machine learning based ones as they time consuming and not efficient.
            - Make sure your solutions handle requirements and conditions of the problem inherently and directly.
            - Avoid using fallbacks as much as you can but in special caases use a decent one and not just a naive version for bypassing errors.
            - There is not limit on the number of lines of code and solution length. Write as much as you can to get the highest score.

            # Error handling requirements:
            - When handling time limit errors follow below steps are critical and must be followed:
                -- Normally the best way is to make your code time aware and run as long as possible without getting TLE.
                -- First completely understand the time complexity of code. 
                -- The first step of error handling must always be using more efficient algorithms and data structures.  
                -- Find bottlenecks and use caching, memoization, precomputing to reduce runtime .
                -- Finally if nothing works you may relax the parameters (but not aggressively) of the algorithm to make it fit within the time limit. however, in this case use the highest parameter that fits within time limit to avoid sacrificing performance.
                -- Sometimes good solutions are implemented in a bad way that results in time limit. make sure to not assume they are bad and prune them just because of a few failed implementations.
            # Output:
            - Your final output must always be a single file named \"main.cpp\" that implements the solution in cpp23 language.
            - If neccessary, you can implement only one other cpp file named \"pre_run.cpp\" that is used for any precompuations to add to main.cpp.
                -- In this case always run the pre_run.cpp first before implementing main.cpp.
                -- It is critical that the runtime of pre_run.cpp be no more than 1 minute. Note that main.cpp runtime is provided by the problem and this 1 minute has nothing to do with it.
            \n\n
        """

    def _prepare_code_from_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def stop_condition(self):
        return False

    def _generate_problem_context(self):
        return llm_utils.LLMBackend().llm_completion(
            model="gpt-5",
            messages=[
                {
                    "role": "system", 
                    "content": """
                        You are an world class optimization problem solver and your role is to clean the problem statement. 
                        Requirements:
                        - Remove unimportant links, images, and extra information.
                        - Do not remove any important information.
                        - Do not remove examples of input and output, but you can clean them.
                        - Do not remove contrtaints.
                    """
                },
                {
                    "role": "user", 
                    "content": "Problem statement: " + str(self.session.problem.statement) + "\n\n constraints: \n" + str(self.session.problem.constraints)
                }
            ]
        )

    def get_feedback_on_tests(self, solution, code, test_cases):
        system_prompt = f"""
            You are a grandmaster programmer and algorithm designer. You are given a <problem>, and a <solution> with its <code> solving the <problem>. finally you can see the output of some . 
            You must provide three one sentence feedbacks on the solution based on the outputs, and their score. 
            - Just mention highly critical feedbacks that can improve the solution, not just mediacore ones.
            - Note that this solution already passed the requirements, so your analyzes and feedbacks should not be on time limit or memory limit errors. 
            - Your feedbacks can be in below categories:
                - Low performance on a general type of test cases.
                    -- Under no circumstances your feedback shouldn't be about a single test case. It must involve a category of test cases.
            - In your feedbacks you must always consider a combination of <test cases> with either <solution> or <code> and never rely only on one.
            - You should not provide any solution on how to fix feedbacks. just mention the issue and how it is results in a bad score. 
            The structure of feedbacks are as follows: 
            # Analyze 1: ...
            # Analyze 2: ...
            # Analyze 3: ...
            # Analyze 4: ...
            # Analyze 5: ...

            <feedbacks> 
            Area of improvement 1: ...
            Area of improvement 2: ...
            Area of improvement 3: ...
            </feedbacks>
        """
        # Build test cases string outside f-string (backslashes not allowed in f-string expressions)
        test_cases_str = '\n'.join(
            [f"Test case {i+1}: input: {test_case.input_str[:500]} \n\n output: {test_case.output_str[:500]} \n\n score: {test_case.absolute_score}" for i, test_case in enumerate(test_cases)]
        )
        user_prompt = f"""
            <problem> 
               {self.get_problem_context()}
            </problem>
            <solution>
            {solution}
            </solution>
            <code>
               {code}
            </code>
            <test cases> 
                {test_cases_str}
            </test cases>
        """
        raw_feedbacks = self.llm.llm_completion_with_system_prompt(
            model="gpt-5", system_prompt=system_prompt, user_message=user_prompt
        )
        feedbacks = re.findall(r'<feedbacks>(.*?)</feedbacks>', raw_feedbacks, re.DOTALL)[0]
        return feedbacks

    def _get_domain_knowledge(self):
          return """
            Helpful knowledge if using different approaches:
            - Some usefull algorithms and data structures:
                -- "Linear, Nonlinear, Quadratic, Sequential Quadratic Programming", "Convex Optimization", "Simplex Method", "Interior Point Methods", "Newton’s Method", "L-BFGS", "Conjugate Gradient", "Subgradient Methods", "Proximal Methods", "Frank–Wolfe Algorithm", "Trust Region Methods", "Augmented Lagrangian Methods", "Alternating Direction Method of Multipliers", "Nelder–Mead",
                -- "Cutting Plane Method", "Column Generation", "Benders Decomposition", "KD-Tree",
                -- "DP", "Branch and Bound","Delta Updates",  "Knapsack", "Set Cover", "Hungarian Algorithm", "Min-Cost Flow", "Dinic’s Algorithm",
                -- "Genetic Algorithm", "Differential Evolution", "CMA-ES", "Simulated Annealing", "Tabu Search", "Particle Swarm Optimization", "Ant Colony Optimization" , "Nested metaheuristic
                -- "Powell’s Method", "Hooke–Jeeves / Pattern Search", "Ellipsoid Method", "Karmarkar’s Algorithm", "Ford–Fulkerson Algorithm", "Edmonds–Karp Algorithm", "Prim’s Algorithm", "Kruskal’s Algorithm", "Firefly Algorithm", "Bat Algorithm"
            - Simulated Annealing:
                -- Simulated annealing is the best approach for optimization problems with possibility of creating a very good first solution and then increamental or local changes. It almost always outperforms simple beam or greedy search.
                -- For better state representation, consider how the current state encoding might be limiting the search space or convergence speed. Think about alternative state encodings that could lead to better local optima or faster convergence.
                -- For better neighborhood design . Consider how the current neighborhood structure might be limiting the search space exploration or convergence speed, and Think about alternative neighborhood structures that could lead to better local optima or faster convergence Specifically , consider :
                    1. How to balance between small and large moves in the search space
                    2. How to ensure the neighborhood structure allows reaching any valid solution
                    3. How to design moves that maintain solution feasibility while exploring new regions
                -- Always think how to push the limit of SA and how to investigate 5x or 10x more valuable states.
                -- Make sure to avoid recomputations for legality check at each step as much as you can, so you can more investigate more neighbors.
                -- In highly constrained problems, keeping the last few steps and having a regret mechanism always helps.
                -- Must consider Adaptive multi phase SA, Metaheuristic SA, SA on hyper parameters, or combining simulating annealings. 
                -- As long as time budget allows, you can run multiple simulated annealings with different initial seed or different state and neighborhood design to take out the maximum efficiency from it.
            - Beam / Random Search / MCTS:
                -- Think about the beam width and evaluation function that could lead to better solutions.
                -- Always consider how to effectively balance between diversity and quality in beams.
                -- Fast stop bad solutions and time control so you can try more solutions as long as time budget allows.
                -- If possible to avoid duplicate states, you can use hashing.
                -- Always think how to push the limits of search and how to investigate 5x or 10x more valuable states.
            - Random simulation:
                -- In non deterministic problems, random simulations are one of the best approaches, specially if problem constraints allow for running highly deep simulations.
                -- Combining creative strategies with random simulation sometimes help for longer horizon and deeper simulations instead of considering a few steps forward.
                -- Depending on the problem balance between next few steps greedy move or longer horizon maximization.
                -- Defining a strong heuristic scoring function is critical for random simulation. Generally it is best to consider both average and std of the scores and run enough simulations to get a good estimate.
                -- Here again avoiding recomputations and fast stopping bad solutions is helpful to run more simulations as much as time budget allows.
            - Ant Colony Optimization (ACO):
                -- consider Ant Colony Optimization (ACO) for Global Plan Construction. population-based metaheuristic approach highly effective for discovering complex inter-dependencies between leg strategies.
            - Parameter and configs:
                -- To avoid overfitting or falling into the local optimum trap, based on problem needs, always consider using dynamic hyperparameters that changes according to the situation.
                -- The parameters must provide a balance between diversity and the quality of the search to find the best answer. One good way is tuning them as you search for the answer. 
            - Conditional Solution Generation:
                -- Sometimes combining multiple approaches helps, for example using different types of Simulated Annealings or the multi phase version for a problem or multiple beam search types based on the problem conditions or iterations. 
                -- Sometimes for different conditions of the problem inputs, it is better to generate different solutions. For example for some problems you can generate different solutions for different input sizes, for example running an efficient brute force in small test cases.
                -- Sometimes for some conditions of the problem you can precompute answers, for example, this can be helpful in problems with fixed and low variant inputs. Make sure to consider this option if it is applicable.
                -- If possible consider hardcoding the answer of some conditions of the problem inputs inside the code.
                -- Always consider multi strategy solutions. but make sure to remove strategies that are not performing well and only waste the runtime capacity.
                --  Some algorithms can constantly be improved and have lots of diversity in their usage like SA and Random Search. 
                    -- Sometimes they might not perform well in a specific type of problem, so they should be combined with another efficient algorithm.
                    -- Sometimes they may face cold start but with constant improving, they mostly perform better in long run.
            - Implementation: 
                -- For operations that are used mostly always make sure to implement the light version. For example simple fast random function instead of rng for time effiecieny.
                -- Make sure to avoid memory allocation in loops.
            - Diversity:
                -- You have access to the summary of previous experiments, make sure consider other than them and try crazy and out of the box ideas and solutions as well or diverse combination and modifications of them.
                -- Beside improving and modification of current ideas mentioned in the experiments summary, Consider generating and selecting new and out of the box core ideas not in the summary for exploration.
                -- Try diverse variations of algorithms and structures. Do not drop any idea or solution just because of a few failed implementations.
        """
