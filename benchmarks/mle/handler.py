from codeop import CommandCompiler
import os
import shutil
from signal import valid_signals
import subprocess
import sys
import re
import tarfile
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import mlebench
from mlebench.data import download_and_prepare_dataset
from mlebench.grade import validate_submission, grade_csv
from mlebench.registry import registry

from kapso.environment.handlers.base import ProblemHandler, ProblemRunResult
from kapso.core import llm as llm_utils

CUDA_DEVICE = int(os.getenv('CUDA_DEVICE', '0'))
MLE_SEED = int(os.getenv('MLE_SEED', '1'))

RUNTIME_LIMIT = 8*60*60
DEBUG_RUNTIME_LIMIT = 20*60

class MleBenchHandler(ProblemHandler):
    def __init__(self, competition_id="spooky-author-identification", fetch_huggingface_models: bool = True):
        # No additional context needed - problem context is dynamically generated
        super().__init__(additional_context="")
        self.llm = llm_utils.LLMBackend()
        self.competition_id = competition_id
        self.problem_id = competition_id
        self.fetch_huggingface_models = fetch_huggingface_models
        self.data_dir = os.path.abspath(f"tmp/mle/{self.competition_id}/data")
        self._prepare_mle_bench_data(self.competition_id)
        self.maximize_scoring = not (grade_csv(Path("dummy.csv"), registry.get_competition(self.competition_id)).is_lower_better)
        self._set_problem_context()
        self.got_medal = False
        self.current_stage = "FULL TRAINING"
        print(self.get_problem_context())

    def _set_problem_context(self):
        cache_file = f"tmp/mle/{self.competition_id}/context.txt"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f: 
                self.raw_problem_context = f.read()
        else:
            self.raw_problem_context = self._generate_problem_context(str(self._prepare_mle_bench_data(self.competition_id)))
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f: 
                f.write(self.raw_problem_context)
        self.data_size = int(subprocess.check_output(['du', '-sb', self.data_dir]).split()[0]) / (1024 ** 3) if os.path.exists(self.data_dir) else 0.0
        self.raw_problem_context += "\n\nYou are forbidden to use torchaudio, mmcv, openmem, faiss, and pcdet as they are incompatible."
        self.problem_context = self.raw_problem_context + f""" \n\n\n
            # Problem Requirements:
            - you are participating in a kaggle competition and your goal is to solve the competition problem and get the best ranking.
            - In this problem the {"higher" if self.maximize_scoring else "lower"} score is better.
            - All the relevant training and test data to the problem are in the {self.data_dir} directory. under no circumstances apply any change to this directory.
            - You must implement a main.py file that running it writes the final_submission.csv file in the Experiment Data Directory. the final_submission.csv file must be a csv for test.csv and test data. a small example of the output can be found on sample_submission.csv.
            - Read and understand the data completely structure completely, and note ther might be some missed files or data in some cases.
            - Do not run the code directly, he should only write the code.

            # Coding style:
            - At the start of the code you must install the necessary packages to run the code if not already installed. Do not show any output and log of the installation progress. Do not reinstall existing packages.
            - Your code must be in python, highly structured and modularized and in more than one file that each one has no more than 100 lines of code.
            - Always first run the code in debug mode to check if the code is working correctly and then run the code in full mode. 
                -- It is crucial to test every part of the code in debug mode so do not discard any part in it, even writing final_submission.csv at the end.
            - Your code must print your steps and progress each part like epochs in the code, and the performance of each part of pipeline, for example the accuracy of each fold or each model in ensembling.
            - ** under no circumstances print active outputs like tqdm, progress bar, etc. Disable the warnings must be the first thing you do at the beginning of the code and in the main file. **
            - Your code must be completely time aware since it will be stopped after 3 hour for small datasets (< 1 GB) and 8 hours for big datasets. 
                -- Always have a timer to measure the time taken for each part of the code and print it.
                -- Use the size and number of data files and folders, and the model to have a better estimate on run time.
            
            # Debug mode
                - Your code must support a debug mode that runs the code with a small proportion of the data (no more than 100 instances but containing all outputs types, for example startified sampling and at least 5 instace from each classes and if too many classes and labels exists (for example more than 500) use a small subset of these classes (at most 100 classes) in debug and all time consuming parameters must be decreased in debug mode. for example no more than 1 epoch in debug mode.
                    -- in debug mode sample before train validation split.
                    -- Running debug mode will be like 'python main.py --debug'
                    -- Running full mode will be like 'python main.py'
                    -- ** make sure to test everything from EDA, to training and even running predictions on the final_submission.csv in the debug mode**.
                    -- Highly CRITICAL: ** In non debug mode, the code must rewrite the final_submission.csv file with the actual predictions. In other words every time you run main.py (both debug mode and non debug mode) it must write final_submission.csv file in the similar and expected path in the Experiment Output Data Directory.
                    -- ** Debug mode must run the whole pipeline for 100 of test set instances and fill other ones with a default value without running models so you can create a valid final_submission.csv file fast. ** 
                    -- make sure that when debug mode is off, all instances of test must be filled with the actual predictions while when it is on only a sample of them are predicted. (This is the most critical part of you job and everything depends on creating a correct final_submission.csv file when debug mode is off, be very careful with this when writing or debugging the code).
        
            # EDA: 
            - Print two eda parts one at the beginning of the code and one at the end of the code.
            - At the beginning of the code you must have an EDA section that describes the data, distributions, features and the labels and information about some samples (not just ids).
                -- Always print a few columns sample submission file to understand the output format.
                -- Print size, shape, and content of a few rows.
                -- Print the content of Problem Data Directory. 
                -- Print anything specific to the problem that can help to better understand the problem for future solutions.
                -- Print anything suspiceous based on previous experiments outputs.
                -- Print any kind of information that can be important in input data, For example, token distribution, image shapes distribution, audio lenght and etc.
            - At the end of the code you must have an error analysis section, It should have anything that can be helpful for future analysis:
                -- In case of ensembling, you must print the accuracy of each model and the ensemble accuracy.
                -- Prediction distribution must be printed.
                -- Incorrect and bad predictions must be printed but not just ids, some information about their features and outputs.
                    --- For images and audios, the average, min, max, median of pixels or data can be the mentioned information.
                -- Always print a samples of validation prediction and validation true label.
                -- Always print a few samples of final_submission.csv
                -- print confusion matrix in cases that it helps.
            - Print any information that can help to find errors in model. For example top 5 important features (specially in tree classifiers) for understanding train validation leakage and overfitting, candidate features to add and etc.
            - Do not consider EDAs as something important that should be improved it based on the recent abd best experiments and information every time. In other words when generating solutions, always think on how to add items and improve the EDA of parent solution and previous experiments for finding issues and better solutions.

            # ML requirements:
            - When generating new ideas and solutions, consider the output of previous experiments and use it to improve them and address their issues.
            - When generating new ideas and solutions, **if the parent node is experimented**, always one of the new nodes must be the same solution with significantly better hyperparameters that have potential to improve the performance meaninfully.
            - Make sure to read the knowledge base and use it. Its knowledge is very important and can help you to solve the problem.
            - Always avoid running with OOF, cross validation & heavy folding, and multi seeds running as they are time consuming and doesn't help to the performance.
            - It is critical to always consider early stopping and always have schedule for learning rate if using neural network models since this helps to converge to the optimal solution in lower epochs. and consider more descriptive metric for early stopping, for example loss is better than accuracy.
            - Make sure to use more well known libraries and models like torch (do not use tensorflow and keras), transformers, sklearn, xgboost and catboost.
                -- Never use LightGB unless working with sparse matrix of sklearn.
                -- Never and never train a heavy model from scratch. always find a way to load pretrained weights.
            - It is ok to put small pieces of codes in the solution generation for the better understanding of coder when helpful or necessary.
            - When splitting train and validation make sure about any kind of leakage. Time-based leakages, content, similar noise, similar type of content (for example similar images and similar speaker voices in train and test), or make sure the models are prune to this kind of leakage. 
            - Use Large and strong models in ensemble and train them long enough with early stopping. finally use a meta-model that is robust to overfitting. in MINI Training stage, avoid heavy ensemble, start with only one large and strong model.
            - One of the most important and highly critical requirements is stratified, smartly and problem aware separatining 10% of the training data for test and train nothing on it. not even hyperparameters. at the end of code report only a single float which is the evaluation metric and score for this test set inside <score> and </score> tags. e.g. <score>0.96</score>.

            # STAGE
                - You have below stages make sure to act accordingly as provided in your solution generation. Note that the default STAGE is "FULL TRAINING".
                    -- STAGE "MINI TRAINING": Startified sampling {round(min(10/self.data_size*100, 10), 1)}% of training dataset and running only on it. Also make sure to take at least 5 sampels from each class (if more than 2000 classes, do not choose 5 from all, sample 1000 classes randomly and choose from them but make sure revert to full classes in FULL TRAINING) and at least 100 total samples.
                        -- This STAGE is used at early stages of experimentation for faster exploration only and only in big datasets (bigger than > 30 gb).
                    -- STAGE "FULL TRAINING": Training on the Whole dataset.
                        -- This Stage is the default stage of training. working on the whole dataset.
                        -- If the dataset is big (> 30 GB) and the Stage changed from MINI to FULL training, make sure to rethink any parameter, hyperparameter, data cleaning, and any other configuration that may affect the performance.
                    -- STAGE "FINAL ENSEMBLING": Ensembling the best models and solutions.
                        -- Search through previous experiments and solutions and take all the best models and parts of their solution and ensemble them. 
                        -- This stage is used when the budget progress is more than 80% and for the final exploitation to generate the final best solution.
                        -- In this stage you may ensemble up to 5 models.
                - Note that in both stages you should keep test set and final submission full.
                - It is critical to mention the sampling of only the current STAGE in the solution generation hyper parameters.
                - Note that this is sampling is different from debug mode and both should be mentioned and both must be in an startified manner.
                - when sampling make sure to shuffle data before.
                - Make sure to consider these stages in your pruning, solution generation, selection. For example dont prune efficient solutions in MINI stage while their score may be low compared to FULL ones.
                - When the STAGE changes make sure to consider the fact that scoring is changed, also make sure to generate solutions and select solutions that performed well the in the previous and exploration stage.
                - Your code must have a hyper parameter STAGE that handles the sampling, note that it is not handled by arguments and it is hardcoded on top of the code. you should change it in the code according to the problem needs.

            # IO
            - You have access to three directories:
                -- Problem Data Directory: which keeps all the data (trian, test, additional) relevant to the problem and should not be changed.
                    -- all zip files are already extracted in this directory.
                    -- Below you can find the content of this directory: 
                    \n\n{self._get_data_structure_string(Path(self.data_dir))}\n\n
                -- Root Directory of git: You must implement your codes here.
                -- Experiment Output Data Directory: Everything that the implemented code writes (final_submission, checkpoints, processed data) should be in this direcory.
            - It is of utmost importance and highly critical to completely control output for avoiding printing any warning or redundant output.
            - Print the time for any time consuming stages , for example after each epoch when training big datasets.
            - When doing a process like loading, caching, or etc. make sure to avoid printing progress bar and set verbose to False and 0 in these cases.
            - It is critical to be careful when reading and writing non-english characters from train, test and csv data. for example use encoding "utf-8-sig" and follow sample submission about their exact encoding and format.

            # Resources
            - You have access to one h100 gpu with 75 gb memory, 30 cpus and 300 gb of ram. 
                -- it is critical to utilize these resources efficiently. consider these in your code. fully utilize gpu, and num_workers and n_jobs to 30.
                    -- For batch size based on data and model decide it during code to take full utilization of gpu and its memory, but make sure to set the epoch size accordingly.
                -- Make sure to set batch size in a way to maximize gpu memory usage but not overflow it. For example start with a big batch size and if get an out of gpu memory error, in a try and catch reduce batch size and try again but never switch to cpu for training big models like neural networks and transformers.
                -- If using GPU make sure to set the default cuda device as \" device = torch.device("cuda:{CUDA_DEVICE}"), torch.cuda.set_device(device)\".
                -- For training transformers and neural networks make sure you are using gpu to its fullests. Under no circumstance train on cpu for these heavy models.
                -- It is critical and of utmost importance to never use any gpu for xgboost, catboost, and lightgbm and always use cpu.  
                -- make sure to empty gpu cache after every run torch.cuda.empty_cache().
        """ + (f"""
            # Hugging Face Models and Datasets:
            Consider these pretrained models and datasets from huggingface if helpful otherwise ignore completely ignore them:\n\n{self._get_models_n_datasets_from_huggingface()}\n\n
            - If dataset is useful for augmentation, do not limit training to its train set and use the test set for augmentation too.
            - Note that you should ensemble these models with mentioned models in the knowledge base and other models as well.
        """ if self.fetch_huggingface_models else "")

    
    def run(self, file_path, run_data_dir, debug=False, *args, **kwargs):
        run_had_error = False
        error_message = ""
        error_details = ""
        output = ""
        score = -1
        continue_debugging = True
        competition = registry.get_competition(self.competition_id)
        submission_path = Path(run_data_dir) / "final_submission.csv"

        try:
            print('#'*100, "Start running the main in debug mode")
            run_had_error, error_details, output, execution_time = self._run_command(file_path, ['python', 'main.py', '--debug'], DEBUG_RUNTIME_LIMIT)
            
            if execution_time + 1 >= DEBUG_RUNTIME_LIMIT:
                run_had_error = True
                error_details = f"Debug execution took {execution_time:.2f} seconds (exceeded 10 minute limit). This solution is not time efficient."
                continue_debugging = False
            if not run_had_error:
                is_valid, error_details = validate_submission(submission_path, competition)
            else:
                is_valid = False
            if not run_had_error and is_valid:
                print('#'*100, "Start running the main in full mode")
                if os.path.exists(submission_path):
                    os.remove(submission_path)
                run_had_error, error_details, output, execution_time = self._run_command(file_path, ['python', 'main.py'], RUNTIME_LIMIT)
                if output and not run_had_error:
                    output = self._clean_run_output(output)
                if run_had_error and execution_time >= RUNTIME_LIMIT:
                    error_details = f"Exceeded {RUNTIME_LIMIT/60} minute limit. This solution is not time efficient."
                    continue_debugging = False

        except Exception as e:
            run_had_error = True
            error_details = str(e)
            if len(error_details) > 1000:
                error_details = error_details[:100] + "..." + error_details[-500:]
            output = ""
        
        if not run_had_error:
            is_valid, error_details = validate_submission(submission_path, competition)
            run_had_error = not is_valid
            if is_valid:
                grading_results = grade_csv(submission_path, competition)
                print(str(grading_results))
                match = re.search(r'<score>(.*?)</score>', output)
                if match:
                    score = float(match.group(1))
                else:
                    score = -1e3 if self.maximize_scoring else 1e3
                self.got_medal = self.got_medal or grading_results.any_medal
                if self.got_medal:
                    print("Got a medal!")
                    dest_dir = Path(os.path.expanduser(f"~/mle_res/{MLE_SEED}/{self.competition_id}"))
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / "final_submission.csv"
                    shutil.copy2(str(submission_path), str(dest_path))
                    
        return ProblemRunResult(
            score=score,
            output=output + f"\n\nCurrent Stage of is {self.current_stage}",
            run_had_error=run_had_error,
            error_message=error_message,
            error_details=error_details,
            detailed_output="",
            feedbacks="",
            continue_debugging= continue_debugging,
        )
        
    def final_evaluate(self, file_path, *args, **kwargs):
        competition = registry.get_competition(self.competition_id)
        submission_path = Path(file_path) / "final_submission.csv"
        return str(grade_csv(submission_path, competition))
    
    def get_current_stage(self, budget_progress):
        if budget_progress < 20 and self.data_size > 30:
            self.current_stage = "MINI TRAINING"
            return "MINI TRAINING"
        elif budget_progress > 80:
                self.current_stage = "FINAL ENSEMBLING"
                return "FINAL ENSEMBLING"
        else:
            self.current_stage = "FULL TRAINING"
            return "FULL TRAINING"

    def get_problem_context(self, budget_progress=0, *args, **kwargs):
        print(f"Current Stage with budget progress {budget_progress} is {self.get_current_stage(budget_progress)}")
        current_problem_context = self.problem_context + f"** Current Stage is {self.get_current_stage(budget_progress)} **\n"
        return current_problem_context

    def stop_condition(self):
        return self.got_medal

    def _run_command(self, file_path, command, timeout=RUNTIME_LIMIT):
        run_had_error = False
        error_details = ""
        output = ""
        start_time = time.time()
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(
                command,
                cwd=file_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,                
            )
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            sys.stdout.flush()
            output_lines.append(line)
            if len(output_lines) > 25000:
                process.kill()
        try:
            process.wait(timeout) 
        except subprocess.TimeoutExpired:
            process.kill()
            print("Process timed out and was killed.")
        end_time = time.time()
        execution_time = end_time - start_time
        if len(output_lines) > 400:
            output_lines = output_lines[:200] + [" ...\n"] + output_lines[-200:]
        output = ''.join(output_lines)
        if process.returncode != 0:
            run_had_error = True
            error_details = output
        return run_had_error, error_details, output, execution_time

    def _clean_run_output(self, output):
        lines = output.split('\n')
        cleaned_lines = [line for line in lines if 'warning' not in line.lower() and 'warn' not in line.lower() and 'info' not in line.lower()]
        no_warning_output =  '\n'.join(cleaned_lines)
        return self.llm.llm_completion(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a world class kaggle problem solver and your role is to clean the output of the code.
                    Requirements:
                    - Remove all warnings, errors, and any other information that is not informative and helpful.
                    - Remove repeating warnings or lines with [INFO] tag. 
                    - Do not remove any important information.
                    - Do not change or remove EDA, training information, error analysis, etc.
                    - Do not remove information about parameters, metrics, results and hyper parameters.
                    - Do not remove the paths to the train, test and sample_submission files.""",
                },
                {
                    "role": "user",
                    "content": "Clean the following output: " + no_warning_output,
                }
            ]
        )

    def _remove_links(self, text):
        if not text:
            return text
        text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]+', '', text)
        text = re.sub(r'www\.[^\s<>"{}|\\^`\[\]]+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'<a[^>]+>([^<]*)</a>', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'href=["\'][^"\']+["\']', '', text, flags=re.IGNORECASE)
        return text

    def _prepare_mle_bench_data(self, dataset_id):
        data_dir_path = Path(self.data_dir)
        data_dir_path.mkdir(parents=True, exist_ok=True)

        competition = registry.get_competition(dataset_id)
        download_and_prepare_dataset(competition)
        all_data_files = list(competition.public_dir.glob('*'))
        
        for file in all_data_files:
            dest_path = data_dir_path / file.name
            if not dest_path.exists():
                if file.is_dir():
                    shutil.copytree(str(file), str(dest_path))
                else:
                    shutil.copy(str(file), str(dest_path))
        
        for archive_path in data_dir_path.glob('*'):
            if archive_path.is_file():
                if archive_path.suffix == '.zip':
                    try:
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            members = zip_ref.namelist()
                            if members:
                                extract_dir = data_dir_path / archive_path.stem
                                extract_dir.mkdir(exist_ok=True)
                                if not (extract_dir / members[0]).exists():
                                    zip_ref.extractall(extract_dir)
                    except zipfile.BadZipFile:
                        pass
                elif archive_path.suffix == '.tar' or '.tar' in archive_path.suffixes or archive_path.suffix == '.tgz':
                    try:
                        with tarfile.open(archive_path, 'r:*') as tar_ref:
                            members = tar_ref.getnames()
                            if members:
                                extract_dir = data_dir_path / archive_path.stem
                                extract_dir.mkdir(exist_ok=True)
                                if not (extract_dir / members[0]).exists():
                                    tar_ref.extractall(extract_dir)
                    except tarfile.TarError:
                        pass        
        
        return competition.description

    def _generate_problem_context(self, problem_description):
        return llm_utils.LLMBackend().llm_completion(
            model="gpt-5",
            messages=[
                {
                    "role": "system", 
                    "content": """
                        You are an world class kaggle problem solver and your role is to clean the problem statement. 
                        Requirements:
                        - Remove all links, hrefs, hyperlinks, and timeline related information.
                        - Remove all unimportant sections like citation and prize and extra information.
                        - Keep any information about the input and output of the problem without any change.
                        - Do not remove examples of input and output, but you can clean them.
                        - Do not remove contrtaints.
                        - Do not remove the paths to the train, test and sample_submission files.
                    """
                },
                {
                    "role": "user", 
                    "content": "Problem statement: \n\n" + problem_description
                }
            ]
        )
    def _get_data_structure_string(self, data_dir_path):
        if not data_dir_path.exists():
            return ""
        
        lines = []
        lines.append(f"Data Directory Structure: {data_dir_path}")
        lines.append("=" * 80)
        lines.append("")
        self._add_directory_contents(lines, data_dir_path, depth=0, max_depth=4, is_last=True, prefix="")
        return "\n".join(lines)
    
    def _format_size(self, size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def _get_directory_size(self, dir_path):
        total_size = 0
        try:
            for item in dir_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except (PermissionError, OSError):
            pass
        return total_size
    
    def _add_directory_contents(self, lines, dir_path, depth, max_depth, is_last=False, prefix=""):
        if depth > max_depth:
            return
        
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        
        dirs = [d for d in items if d.is_dir()]
        files = [f for f in items if f.is_file()]
        
        shown_files = files[:5]
        shown_dirs = dirs[:5]
        remaining_files = len(files) - len(shown_files)
        remaining_dirs = len(dirs) - len(shown_dirs)
        
        has_more_files = remaining_files > 0
        has_more_dirs = remaining_dirs > 0
        has_dirs = len(shown_dirs) > 0
        
        for i, file_path in enumerate(shown_files):
            is_last_file = (i == len(shown_files) - 1) and not has_more_files and not has_dirs
            connector = "└──" if is_last_file else "├──"
            file_size = file_path.stat().st_size
            size_str = self._format_size(file_size)
            lines.append(f"{prefix}{connector} [{size_str}] 📄 {file_path.name}")
        
        if has_more_files:
            connector = "└──" if not has_dirs else "├──"
            lines.append(f"{prefix}{connector} ... ({remaining_files} more file{'s' if remaining_files > 1 else ''})")
        
        for i, dir_item in enumerate(shown_dirs):
            is_last_dir = (i == len(shown_dirs) - 1) and not has_more_dirs
            connector = "└──" if is_last_dir else "├──"
            dir_size = self._get_directory_size(dir_item)
            size_str = self._format_size(dir_size)
            lines.append(f"{prefix}{connector} [{size_str}] 📁 {dir_item.name}/")
            
            next_prefix = prefix + ("    " if is_last_dir else "│   ")
            self._add_directory_contents(lines, dir_item, depth + 1, max_depth, is_last=is_last_dir, prefix=next_prefix)
        
        if has_more_dirs:
            lines.append(f"{prefix}└── ... ({remaining_dirs} more director{'ies' if remaining_dirs > 1 else 'y'})")

    def _get_models_n_datasets_from_huggingface(self):
        """
        Fetch relevant models and datasets from HuggingFace using web search.
        Uses unified LLM layer with web search support.
        """
        prompt = f"""
            You are a world class kaggle problem solver and your role is to get the models and datasets from huggingface by web search.
            Requirements:
            - Do not provide any datasets if the the current dataset size is bigger than 5GB. Current dataset size is {self.data_size} GB.
            - search for the models and datasets from huggingface for the given problem.
            - Return the models and datasets in a dictionaries.
            - Note that if you can't find any related model or dataset, return an empty dictionary. among all the available models and datasets, return the most similar ones to the problem.
            - Do note write anything extra, just the dictionary.
            - Provide at least 2 and at most 5 models and 1 dataset. These models and datasets must be specific and the most similar ones to the problem. If none found return the most downloaded models in the domain of problem (e.g. speech recognition, text generation, etc.)""" + """
            - Note that model size should not be bigger than 10 B parameters.
            - Do not drop a dataset if it is for a model and vice versa. make sure to return both if relevant to the the problem.
            - Search throughly and make sure you are not missing any model or dataset.
            - Search git repos and kaggle for any winning method and solution similar to the problem. After similarity to the problem prioritize winners.
            - The final output must be in the output format below. return empty list if none found.
            Output format:
            {
                "models": [
                    {
                        "name": "model_name",
                        "description": "summary of the model",
                        "number_of_parameters": number of parameters of the model,
                        "usage": "python code of usage",
                    },
                    {
                        "name": "model_name",
                        "description": "summary of the model",
                        "number_of_parameters": number of parameters of the model,
                        "usage": "python code of usage",
                    }
                ],
                "datasets": [
                    {
                        "name": "dataset_name",
                        "description": "dataset_description",
                    },
                    {
                        "name": "dataset_name",
                        "description": "dataset_description",
                    },
                ]
            } \n """ + f"Problem statement: \n\n + {self.raw_problem_context}"
        
        messages = [{"role": "user", "content": prompt}]
        
        # Use unified LLM layer with web search - parallel calls with different reasoning efforts
        responses = self.llm.llm_multiple_completions_with_web_search(
            models=["gpt-4o-search-preview", "gpt-4o-search-preview", "gpt-4o-search-preview"],
            messages=messages,
            search_context_size="high",
            reasoning_efforts=["medium", "medium", "high"],
        )

        cleaned_output = "".join([self._remove_links(r) for r in responses])
        
        # Final cleanup call (no web search needed)
        final_response = self.llm.llm_completion(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user", 
                "content": cleaned_output + """
                clean the above outputs, remove repeating models, datasets but keep everything else specially content. 
                your final output must be a single dictionary like below:
                {
                    "models": [
                        {
                            "name": "model_name",
                            "description": "summary of the model",
                            "number_of_parameters": number of parameters of the model,
                            "usage": "python code of usage",
                        },
                        {
                            "name": "model_name",
                            "description": "summary of the model",
                            "number_of_parameters": number of parameters of the model,
                            "usage": "python code of usage",
                        }
                    ],
                    "datasets": [
                        {
                            "name": "dataset_name",
                            "description": "dataset_description",
                        },
                        {
                            "name": "dataset_name",
                            "description": "dataset_description",
                        },
                    ]
            """
            }]
        )

        return final_response   
    
if __name__ == "__main__":
    handler = MleBenchHandler()
    handler._get_models_n_datasets_from_huggingface()

