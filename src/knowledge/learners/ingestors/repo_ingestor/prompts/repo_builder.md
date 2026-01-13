# Repository Builder Phase

You are responsible for creating a GitHub repository from pre-generated workflow files and pushing them.

## Context

You have been given:
- **Generated files directory**: `{files_dir}` - Contains all workflow implementation files
- **Suggested repo name**: `{repo_name}` - The preferred GitHub repository name
- **Workflow name**: `{workflow_name}` - For reference/description
- **Workflow description**: `{workflow_description}`
- **Visibility**: `{visibility}` - Either "private" or "public"
- **Result file**: `{result_file}` - Where to write the final GitHub URL

## Your Task

1. **Check if repository already exists**
2. **Create the repository** (with alternative name if needed)
3. **Initialize git, commit, and push the files**
4. **Write the final GitHub URL to the result file**

## Step-by-Step Instructions

### Step 1: Verify GitHub CLI Authentication

First, verify that `gh` CLI is authenticated:

```bash
gh auth status
```

If not authenticated, the GITHUB_PAT environment variable should be available. You can authenticate with:

```bash
echo $GITHUB_PAT | gh auth login --with-token
```

### Step 2: Check if Repository Exists

Check if the suggested repository name already exists:

```bash
gh repo view {repo_name} --json name 2>/dev/null && echo "EXISTS" || echo "AVAILABLE"
```

### Step 3: Choose Repository Name

- If the name is **AVAILABLE**: Use `{repo_name}`
- If the name **EXISTS**: Try alternatives in order:
  1. `{repo_name}-v2`
  2. `{repo_name}-v3`
  3. `{repo_name}-v4`
  4. Continue until you find an available name

Check each alternative:
```bash
gh repo view {repo_name}-v2 --json name 2>/dev/null && echo "EXISTS" || echo "AVAILABLE"
```

### Step 4: Create the Repository

Once you have an available name, create the repository:

```bash
gh repo create {final_repo_name} --{visibility} --description "{workflow_description}" --source={files_dir} --push
```

This command:
- Creates a new repository
- Sets visibility (private/public)
- Uses the generated files as source
- Automatically initializes git, commits, and pushes

**If the above command fails**, fall back to manual steps:

```bash
cd {files_dir}
git init
git config user.email "praxium-bot@example.com"
git config user.name "Praxium Bot"
git add .
git commit -m "Initial workflow implementation"
git branch -M main
gh repo create {final_repo_name} --{visibility} --source=. --remote=origin --push
```

### Step 5: Get the Repository URL

After creation, get the full URL:

```bash
gh repo view {final_repo_name} --json url -q .url
```

### Step 6: Write the Result

Write the final GitHub URL to the result file:

```bash
echo "https://github.com/YOUR_USERNAME/{final_repo_name}" > {result_file}
```

Or more precisely:
```bash
gh repo view {final_repo_name} --json url -q .url > {result_file}
```

## Error Handling

### Authentication Error
If you get authentication errors, ensure GITHUB_PAT is set:
```bash
echo "GITHUB_PAT is set: $([ -n "$GITHUB_PAT" ] && echo 'yes' || echo 'no')"
```

### Name Conflict
If all version suffixes are taken (v2 through v10), use a timestamp suffix:
```bash
{repo_name}-$(date +%Y%m%d)
```

### Push Error
If push fails with "src refspec main does not match any":
```bash
git branch -M main
git push -u origin main
```

## Success Criteria

Your task is complete when:
1. ✅ A GitHub repository is created (private or public as specified)
2. ✅ All workflow files are pushed to the repository
3. ✅ The result file contains the full GitHub URL (e.g., `https://github.com/username/workflow-name`)

## Example Session

```bash
# Check auth
gh auth status

# Check if name exists
gh repo view workflow-unsloth-qlora-finetuning --json name 2>/dev/null && echo "EXISTS" || echo "AVAILABLE"
# Output: AVAILABLE

# Create and push
gh repo create workflow-unsloth-qlora-finetuning --private --description "QLoRA fine-tuning workflow" --source=/tmp/praxium_workflow_xyz --push

# Write result
gh repo view workflow-unsloth-qlora-finetuning --json url -q .url > /path/to/result.txt
```
