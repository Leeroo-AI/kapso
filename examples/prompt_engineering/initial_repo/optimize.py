"""
Prompt template for solving AIME (American Invitational Mathematics Examination) problems.

This file contains the prompt that will be optimized by Kapso to improve
accuracy on AIME math problems.
"""

from openai import OpenAI

client = OpenAI()  # API key must be in OPENAI_API_KEY

# The prompt template that instructs the LLM how to solve AIME problems.
# Kapso will optimize this to improve accuracy.
PROMPT_TEMPLATE = """You are an expert competition mathematician tasked with solving an AIME problem.
The final answer must be a three-digit integer between 000 and 999, inclusive.
Please reason step-by-step towards the solution. Keep your reasoning concise.
Conclude your response with the final answer enclosed in \\boxed{{}}. For example: The final answer is \\boxed{{042}}.

Problem:
{problem}

Solution:
"""


def solve(problem: str, model_name: str) -> str:
    """
    Return the model's raw text answer for one problem using the specified model.
    
    Args:
        problem: The AIME problem text to solve.
        model_name: The OpenAI model to use (e.g., "gpt-4.1-mini").
    
    Returns:
        The model's response text containing the solution and answer.
    """
    prompt = PROMPT_TEMPLATE.format(problem=problem)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
