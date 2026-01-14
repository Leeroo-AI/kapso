You are a world class developer. Debug the Implemented <solution> for <problem>.

<repository_memory>
{{repo_memory_brief}}

{{repo_memory_detail_access_instructions}}

OBSERVABILITY REQUIREMENT (do not skip):
- If you consulted repo memory sections, record which ones in `changes.log`.
- Add a line exactly like:
  RepoMemory sections consulted: core.architecture, core.where_to_edit
- If you did not consult repo memory, write:
  RepoMemory sections consulted: none
</repository_memory>

<problem>
{{problem}}
</problem>

<solution>
{{solution}}
</solution>

current output: {{error_details}}

Requirements:
- Read the code line by line and understand the logic.
- Make sure every part of the <solution> is implemented correctly.
  - Read sections and steps of <solution> carefully and implement them exactly.
- Do not propose a new solution or drift away from the current implementation for <solution> and only fix the errors.
- Write clean, functional code, that can be improved iteratively later.
- Output code and format must be as mentioned in the problem statement.
- Do not add logics like fallback and functionality discarding to avoid the error. you must fix the error directly.
- Never and under no circumstances use try except blocks to fix the errors. you should fix the error directly.
- Beside fixing the current error, read the code and make sure other parts of the code will be run correctly and without errors.
- Do not change any hyper parameter or logic of the solution to fix the error.
- Do not ask any questions from the user. just do as you said.

