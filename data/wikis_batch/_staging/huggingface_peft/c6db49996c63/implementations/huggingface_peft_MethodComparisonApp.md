# Method Comparison App - Implementation

## Overview

The Method Comparison App is a Gradio-based interactive visualization application for comparing Parameter-Efficient Fine-Tuning (PEFT) methods. It provides comprehensive visual analysis of experimental results through Pareto frontier plots, allowing users to identify optimal PEFT configurations based on multiple performance metrics.

**File Location:** `/tmp/praxium_repo_35tl5_4u/method_comparison/app.py`

**Related Files:**
- `/tmp/praxium_repo_35tl5_4u/method_comparison/processing.py` - Data loading and preprocessing
- `/tmp/praxium_repo_35tl5_4u/method_comparison/sanitizer.py` - Safe query parsing and filtering

## Primary Purpose

This implementation serves as a result dashboard for the PEFT method comparison project. It enables:

1. **Comparative Analysis** - Side-by-side comparison of different PEFT methods across multiple metrics
2. **Pareto Optimization** - Visualization of trade-offs between competing metrics (e.g., accuracy vs. memory usage)
3. **Interactive Exploration** - Dynamic filtering and data export for custom analyses
4. **Decision Support** - Helps users identify the best PEFT method for their specific constraints

## Architecture

### Core Components

#### 1. Data Management
- **Data Loading** (`load_df` from `processing.py`)
  - Loads JSON experiment results from the `MetaMathQA/results` directory
  - Preprocesses data into structured pandas DataFrame
  - Filters to most recent runs per experiment
  - Handles multiple metrics: memory, time, accuracy, loss, model size

- **Data Filtering** (`parse_and_filter` from `sanitizer.py`)
  - Uses AST parsing for safe query evaluation (prevents code injection)
  - Supports comparison operators: `>`, `>=`, `<`, `<=`, `==`, `!=`, `in`, `not in`
  - Boolean logic: `and`, `or`, `not`
  - Example: `peft_type=='LORA' and test_accuracy > 0.8`

#### 2. Pareto Frontier Calculation

The Pareto frontier identifies non-dominated solutions where no other method is strictly better in all compared metrics.

**Algorithm** (`compute_pareto_frontier`, lines 52-87):
```python
def dominates(a, b, metric_x, metric_y):
    # Point b dominates point a if:
    # - b is at least as good as a in both metrics
    # - b is strictly better in at least one metric
```

**Key Features:**
- Respects metric preferences (higher/lower is better)
- Identifies all non-dominated points
- Returns subset of DataFrame with optimal configurations

#### 3. Visualization Engine

**Pareto Plot** (`generate_pareto_plot`, lines 90-156):
- Uses Plotly for interactive graphics
- Three visual layers:
  1. **Blue line** - Connects Pareto frontier points
  2. **Gray markers** - Non-optimal configurations (semi-transparent)
  3. **Colored markers** - Pareto-optimal points (colored by experiment name)
- Hover tooltips show experiment details and metric values

**Plot Configuration:**
- Seaborn theme for professional appearance
- 700px height for readability
- Automatic axis scaling
- Legend for experiment identification

#### 4. User Interface

**Gradio Components:**

1. **Task/Model Selection** (lines 200-209)
   - Dropdown for task selection (e.g., "MetaMathQA")
   - Cascading model dropdown (updates based on task)

2. **Results Table** (lines 217-224)
   - Displays all experimental results
   - Formatted with 3 decimal precision
   - Custom column widths (300px for experiment names, 150px for others)
   - Non-interactive for stability

3. **Dynamic Filtering** (lines 227-233)
   - Text input for filter expressions
   - "Apply Filter" and "Reset Filter" buttons
   - Preserved filter state across interactions

4. **Pareto Analysis** (lines 240-268)
   - Two metric dropdowns for X and Y axes
   - Default: `accelerator_memory_max` vs `test_accuracy`
   - Interactive plot with zoom and pan
   - Summary statistics box

5. **Data Export** (lines 364-371)
   - Export filtered data as CSV
   - Temporary file generation
   - Download through Gradio File component

### Metrics Configuration

**Tracked Metrics** (`metric_preferences`, lines 27-38):

| Metric | Preference | Description |
|--------|-----------|-------------|
| `accelerator_memory_reserved_avg` | lower | Average GPU memory reserved |
| `accelerator_memory_max` | lower | Peak GPU memory usage |
| `accelerator_memory_reserved_99th` | lower | 99th percentile memory |
| `total_time` | lower | Total execution time |
| `train_time` | lower | Training time only |
| `file_size` | lower | Saved model size |
| `test_accuracy` | higher | Test set accuracy |
| `train_loss` | lower | Final training loss |
| `num_trainable_params` | lower | Trainable parameter count |
| `forgetting*` | lower | Knowledge retention metric |

## Key Functions

### `compute_pareto_frontier(df, metric_x, metric_y)`
**Purpose:** Identifies non-dominated solutions in two-dimensional metric space

**Algorithm:**
1. Extract metric values as numpy array
2. For each point, check if any other point dominates it
3. Point is dominated if another point is:
   - At least as good in both metrics
   - Strictly better in at least one metric
4. Return DataFrame subset of non-dominated points

**Returns:** DataFrame containing only Pareto-optimal configurations

### `generate_pareto_plot(df, metric_x, metric_y)`
**Purpose:** Creates interactive Plotly visualization of Pareto frontier

**Visualization Strategy:**
1. Compute Pareto frontier
2. Separate Pareto and non-Pareto points
3. Draw connection line for frontier (sorted by X-axis)
4. Plot non-frontier points in gray
5. Overlay frontier points with color legend

**Returns:** Plotly Figure object

### `compute_pareto_summary(filtered, pareto_df, metric_x, metric_y)`
**Purpose:** Generates statistical summary for selected data

**Output:**
- Min, max, mean for both metrics
- Total data points
- Number of Pareto frontier points
- Number of excluded (dominated) points

### `build_app(df)`
**Purpose:** Constructs complete Gradio application

**Interaction Flow:**
1. User selects task → model list updates
2. User selects model → table filters
3. User applies filter → table, plot, and summary update
4. User changes metrics → plot and summary update
5. User exports → CSV file generated

## Usage

### Running the Application

```bash
# Install dependencies
python -m pip install -r requirements-app.txt

# Launch app
python app.py
```

The app loads results from `method_comparison/MetaMathQA/results` and launches on local web server.

### Example Workflow

1. **Select Task and Model**
   - Choose "MetaMathQA" task
   - Select base model (e.g., "meta-llama/Llama-3.2-1B")

2. **Apply Filters**
   - Input: `peft_type in ['LORA', 'LoKr'] and test_accuracy > 0.75`
   - Click "Apply Filter"

3. **Analyze Trade-offs**
   - Set X-axis: `accelerator_memory_max`
   - Set Y-axis: `test_accuracy`
   - Identify Pareto-optimal configurations (colored points)

4. **Export Results**
   - Click "Export Filtered Data"
   - Download CSV for further analysis

## Integration Points

### Data Source
- Reads JSON files from `MetaMathQA/results` directory
- Each JSON contains:
  - `run_info` - Experiment configuration
  - `train_info` - Training metrics and memory usage
  - `meta_info` - Package and system information

### External Dependencies
- **Gradio** - Web UI framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **processing.py** - Custom data loading pipeline
- **sanitizer.py** - Safe query evaluation

## Implementation Details

### Filter Safety
The application uses AST parsing instead of `DataFrame.query()` to prevent arbitrary code execution:

```python
# Unsafe: df.query(user_input)  # Could execute malicious code
# Safe: parse_and_filter(df, user_input)  # Only allows predefined operations
```

### State Management
- `filter_state` - Hidden Gradio State component tracks current filter
- Preserved across task/model changes
- Allows compound filtering operations

### Column Ordering
Results table prioritizes important metrics:
1. Experiment identification (name, PEFT type)
2. Performance metrics (time, accuracy, loss)
3. Resource metrics (memory, parameters, file size)
4. Metadata (versions, timestamps)

### Responsive Updates
Gradio event handlers chain updates:
- Task change → Model dropdown + Table
- Model change → Table
- Filter change → Table + Plot + Summary
- Metric change → Plot + Summary

## Performance Considerations

### Efficiency Optimizations
1. **Data Loading** - Preprocessed once at startup
2. **Pareto Computation** - O(n²) algorithm acceptable for experimental datasets
3. **Plotting** - Client-side interactivity via Plotly
4. **CSV Export** - Temporary files auto-cleaned by system

### Scalability
- Current implementation handles hundreds of experiments
- For thousands of experiments, consider:
  - Lazy loading strategies
  - Approximate Pareto frontier algorithms
  - Server-side aggregation

## Configuration

### Default Plot Metrics
```python
x_default = "accelerator_memory_max"  # Memory usage
y_default = "test_accuracy"            # Model performance
```

### Theme
```python
demo.launch(theme=gr.themes.Soft())  # Professional, muted color scheme
```

### Table Formatting
```python
column_widths = ["150px" for _ in df.columns]
column_widths[column2index['experiment_name']] = '300px'  # Wider for long names
```

## Extension Points

### Adding New Metrics
1. Add metric to `metric_preferences` dictionary
2. Specify preference ("higher" or "lower")
3. Ensure metric exists in processed DataFrame
4. Available immediately in dropdown

### Supporting Multiple Tasks
Current implementation loads only MetaMathQA. To support multiple:
1. Modify `load_df` calls to load multiple directories
2. Concatenate DataFrames
3. Task dropdown already filters correctly

### Custom Visualizations
The modular design allows adding new plot types:
1. Create new generation function (like `generate_pareto_plot`)
2. Add UI controls (dropdowns, buttons)
3. Wire with Gradio event handlers

## Security Considerations

### Query Sanitization
- AST-based parsing prevents code injection
- Whitelist of allowed operations
- No access to `eval()` or `exec()`

### File System Access
- Reads from predetermined directories only
- Temporary files for exports use secure naming
- No user-controlled file paths

## Deployment

### Local Deployment
```bash
python app.py
# Access at http://localhost:7860
```

### Hugging Face Spaces
The application is deployed at:
- https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison

Configuration in `README.md` header:
```yaml
title: PEFT Method Comparison
sdk: gradio
app_file: app.py
```

## Maintenance Notes

### Data Freshness
- Results updated when new experiments complete
- Most recent run per experiment auto-selected
- Timestamp-based deduplication

### Version Compatibility
- Gradio API may change between versions
- Pin versions in `requirements-app.txt`
- Test after dependency updates

## Limitations

1. **Memory Constraints** - Entire dataset loaded into memory
2. **Two-Metric Pareto** - Only considers two metrics at a time (could extend to 3D)
3. **Static Preferences** - Metric preferences hardcoded (could make user-configurable)
4. **Single Task Loading** - Loads only MetaMathQA by default

## Future Enhancements

1. **Multi-dimensional Pareto** - Support 3+ metrics with dimension reduction
2. **Automated Recommendations** - Suggest best PEFT method based on constraints
3. **Historical Tracking** - Compare experiments over time
4. **Batch Comparison** - Compare multiple task/model combinations
5. **Interactive Tutorials** - Guided tours for new users

## Related Documentation

- [PEFT Method Comparison README](https://github.com/huggingface/peft/tree/main/method_comparison)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Plotly Python](https://plotly.com/python/)

## Summary

The Method Comparison App provides a sophisticated yet user-friendly interface for analyzing PEFT experiment results. Its Pareto frontier visualization effectively communicates trade-offs between competing objectives, enabling data-driven decisions about which PEFT method to use. The combination of interactive filtering, statistical summaries, and data export makes it a comprehensive tool for both quick exploration and deep analysis.
