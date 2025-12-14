# Token-Efficient Context Manager
#
# Context manager that uses accumulated summaries and embedding-based retrieval.
# Registered as "token_efficient" via the factory decorator.

import numpy as np
from scipy.spatial.distance import cosine
from typing import Any, Dict, List, Optional, Tuple

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler
from src.core.llm import LLMBackend


@register_context_manager("token_efficient")
class TokenEfficientContextManager(ContextManager):
    """
    Token-efficient context manager.
    
    Gathers context from multiple sources and enriches it with
    knowledge from an injected search backend (KG, RAG, etc.).
    
    Sources:
    - Problem handler (problem description with budget awareness)
    - Search strategy (experiment history - best and recent)
    - Knowledge search (injected - can be KG LLM Navigation, RAG, or null)
    
    Params (defined in context_manager.yaml):
        - max_experiment_history_count: Max top experiments to include
        - max_recent_experiment_count: Max recent experiments to include
    """
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize token-efficient context manager."""
        super().__init__(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=params,
        )
        self.llm = LLMBackend()
        self.accumulated_summary = ""
        self.accumulated_experiments_count = 0

        self.state_experiments_count = self.params.get("state_experiments_count", 5)
        self.relevant_retrieval_experiments_count = self.params.get("relevant_retrieval_experiments_count", 3)
        self.top_experiments_in_context_count = self.params.get("top_experiments_in_context_count", 2)
        self.recent_experiments_in_context_count = self.params.get("recent_experiments_in_context_count", 2)
        self.summary_token_limit = self.params.get("summary_token_limit", 1000)
        self.summary_model = self.params.get("summary_model", "gpt-5-mini")

    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Gather and enrich context for solution generation.
        
        Args:
            budget_progress: Current budget progress (0-100)
            
        Returns:
            ContextData with problem, history, and knowledge results
        """
        problem = self.problem_handler.get_problem_context(budget_progress=budget_progress)
        kg_results = ""
        kg_code_results = ""
        experiments_context = ""

        all_experiments = self.search_strategy.get_experiment_history()
        print("Preparing token efficient context for problem ...")

        if len(all_experiments) >= self.top_experiments_in_context_count + self.recent_experiments_in_context_count:
            top_experiments = self.search_strategy.get_experiment_history(best_last=True)[-self.top_experiments_in_context_count:]
            recent_experiments = all_experiments[-self.recent_experiments_in_context_count:]
            self._update_accumulated_summary(problem, all_experiments)

            state_embedding = self._get_state_embedding(all_experiments[-self.state_experiments_count:])
            relevant_experiments = self._retrieve_relevant_experiments(state_embedding, all_experiments)

            experiments_context = self._get_summarized_experiments_context(
                self.accumulated_summary, top_experiments, relevant_experiments, recent_experiments
            )
            print("="*100)
            print("Accumulated summary:")
            print(str(self.accumulated_summary))
            print("="*100)
        elif len(all_experiments) != 0:
            experiments_context = (
                "## Recent Experiments:\n" 
                + "\n".join(str(exp) for exp in all_experiments) 
            )

        if len(self.problem_handler.additional_context) > 0:
            kg_results += self.problem_handler.additional_context + "\n\n"

        if self.knowledge_search.enabled and 0:
            print('Searching knowledge graph for context...')
            last_exp_context = str(all_experiments[-1]) if all_experiments else None
            knowledge_result = self.knowledge_search.search(
                query=problem,
                context=last_exp_context,
            )
            if not knowledge_result.is_empty:
                kg_results += knowledge_result.text_results
                kg_code_results = knowledge_result.code_results

        return ContextData(
            problem=problem,
            kg_results=kg_results,
            kg_code_results=kg_code_results,
            additional_info=experiments_context,
        )
    
    def _update_experiment_embeddings(self, experiments: List[Any]) -> None:
        """Update stored embeddings for new experiments."""
        existing_count = len(self.experiment_embeddings)
        new_experiments = experiments[existing_count:]

        for exp in new_experiments:
            embedding = exp.get_embedding(self.llm)
            if embedding:
                self.experiment_embeddings.append((exp, embedding))

    def _update_accumulated_summary(self, problem: str, experiments: List[Any]) -> str:
        """Update accumulated summary with new experiment, respecting token limit."""
        new_experiments_list = experiments[self.accumulated_experiments_count:]
        if len(new_experiments_list) == 0:
            return self.accumulated_summary
        system_prompt = f"""
            You are a world class problem solver.
            You are given a problem, an accumulated summary of previous experiments and a list of new experiments.
            You need to create an updated accumulated summary that incorporates and accumulates the new experiments and preserves the compressed version of most important information, solution, feedback, output and scores from previous experiments. 
            It is critical that your summarization must be so neat and minimal and high quality that it can handle up to 100 experiments. Note that you are not allowed to increase the size of summary it must remain at most 4 or 5 paragraphs.
            - The updated summary should focus on the categories of experiments, patterns, and the most important information, solution, feedback, output and scores from previous experiments.
            - Do not provide repeating information but make sure to cover every key information that helps solving the problem or progressing toward the goal.
            - Your role is not to generate new ideas or solutions, but to compress, find patterns and the most important information from previous experiments.
            - Avoid highly detailed information unless they had meaningful impact on the problem progress, score, output or solution quality.
            - You are not allowed to generate any new information, just extracting and combining and concluding from the previous information.
            - Do not mention any redundant and extra information like repeating problem description or goal. only and only the summary. 
            - You have a {self.summary_token_limit} token limit to generate the accumulated summary so make sure to use it wisely.
            - CRITICAL: It is highly important to keep the summary as minimal as possible. The final goal of this summarization to capture the core and impactful information, not just appending experiments. 
                -- You are not allowed to just appending experiments. make sure to drop redundant and unimportant parts.
                -- You are not allowed to show detailed outputs just distil the knowledge that can be extracted from the outputs and feedbacks.
                -- You are not allowed to suggest solutions and generate new ideas. just witnessed patterns, information and conclusions.
                -- Make sure to mention the categories of working and best performing experiments and a summary of their scores and output info, and whether they have potential to progress toward goal and solving problem.
            - Keep the information about summary highly oraganized. markdown format with the following sections:
                --  High level workflows of experiments and core ideas: categorize experiments into categories based on the idea, solution and steps.
                    --- find patterns and compress and summarize the output, and performance, score and quality of each category. (preferred one sentence but at most 3 sentences for each category)
                    --- menttion specific concepts and parts of the experiments that are important for the category. (preferred one sentence butat most 3 sentences for each part)
                    --- mention any feedbacks and issues that are mentioned in the experiment, note that you should not create feedbacks and you should just aggregate existing ones. preferred one sentence but No more than 3 sentences for eachcategory.
                    --- Details, hyperparameters and specific parts must be mentioned only when have very high on progress toward goal and solving problem. only one sentence.
                -- All in all make sure to combine information and extract high level patterns and low level patterns if impactful. Cover everything while respecting the token limit.
        """
        user_prompt = f"""
            # problem / Goal: 
            {problem}
            # Previous accumulated summary of experiments:
            {self.accumulated_summary if self.accumulated_summary else "No previous accumulated summary"}
            # New experiments:
            {"\n".join(str(exp) for exp in new_experiments_list)}
        """
        
        self.accumulated_summary = self.llm.llm_completion_with_system_prompt(
            model=self.summary_model,
            system_prompt=system_prompt,
            user_message=user_prompt,
        )
        self.accumulated_experiments_count = len(experiments)

    def _get_state_embedding(self, experiments: List[Any]) -> List[float]:
        """Get the embedding of the state by averaging the embeddings of the recent experiments."""
        embeddings = np.array([exp.get_embedding(self.llm) for exp in experiments])
        return np.mean(embeddings, axis=0)/np.linalg.norm(np.mean(embeddings, axis=0))

    def _retrieve_relevant_experiments(self, state_embedding: List[float], all_experiments: List[Any]) -> List[Any]:
        """Retrieve relevant experiments by embedding similarity to the state."""

        similarities = []
        for exp in all_experiments:
            similarity = float(1 - cosine(state_embedding, exp.get_embedding(self.llm)))
            similarities.append((exp, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in similarities[:self.relevant_retrieval_experiments_count]]
    
    def _get_summarized_experiments_context(
        self, 
        accumulated_summary: str, 
        top_experiments: List[Any], 
        relevant_experiments: List[Any], 
        recent_experiments: List[Any]
    ) -> str:
        """Create final experiments context with summaries of accumulated summary, top, relevant, and recent experiments."""
        recent_experiments = [exp for exp in recent_experiments if exp not in top_experiments]
        relevant_experiments = [exp for exp in relevant_experiments if exp not in top_experiments and exp not in recent_experiments]

        final_context = f"## Summary of All Previous Experiments:\n{accumulated_summary}\n\n"

        top_text = "\n\n".join(str(exp) for exp in top_experiments)
        final_context += f"## Top Performing Experiments:\n{top_text}\n\n"

        if recent_experiments:
            recent_text = "\n\n".join(str(exp) for exp in recent_experiments)
            final_context += f"## Recent Experiments:\n{recent_text}\n\n"

        if relevant_experiments:
            relevant_text = "\n\n".join(str(exp) for exp in relevant_experiments)
            final_context += f"## Relevant Experiments (Similar to Current State):\n{relevant_text}\n\n"

        return final_context
