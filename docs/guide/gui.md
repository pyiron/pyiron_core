# PyironFlow GUI: Visual Workflow Development

The PyironFlow GUI provides an intuitive visual interface for designing, executing, and managing scientific workflows. This documentation explains how to use the graphical interface to build and run computational pipelines without extensive coding.

## Getting Started

The simplest way to launch the PyironFlow GUI is with just two lines of code:

```python
import pyiron_core as pc

pf = pc.PyironFlow()
pf.gui
```

This will:

1. Initialize a PyironFlow instance with default settings
2. Launch an interactive graphical interface in your Jupyter Notebook
3. Display an empty workflow canvas ready for node creation

## Configuration Options

The `PyironFlow` constructor accepts several parameters to customize your workflow environment:

### Basic Configuration

```python
pf = pc.PyironFlow(
    wf_list=None,
    hash_nodes=False,
    workflow_path="/path/to/workflows",
    nodes_path="/path/to/nodes"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wf_list` | list | `None` | Predefined list of workflows to load on startup |
| `hash_nodes` | bool | `False` | Whether to use hash-based node identification (improves performance with large workflows) |
| `workflow_path` | str/Path | System-dependent | Directory where workflows are saved |
| `nodes_path` | str/Path | `None` | Directory containing custom node definitions (if different from default) |

### Database Integration

```python
from pyiron_core.pyiron_workflow.graph.gui import create_db

pf = pc.PyironFlow(
    db=create_db()
)
```

The `db` parameter connects PyironFlow to a database for:

- Storing workflow execution history
- Tracking provenance information
- Enabling workflow sharing between users
- Providing persistent storage for results

If you're using pyiron_core's standard database setup, you can use `create_db()` to get the configured database instance.

### Custom Layouts

```python
from pyiron_core.pyiron_workflow.graph.gui import GUILayout

custom_layout = GUILayout(
    flow_widget_width=600,
    flow_widget_height=400,
    output_widget_width=400
)

pf = pc.PyironFlow(gui_layout=custom_layout)
```

The `gui_layout` parameter allows customization of the visual appearance:

- `flow_widget_width`: Width of the main workflow canvas (default: 1200)
- `flow_widget_height`: Height of the main workflow canvas (default: 800)
- `output_widget_width`: Width of the output panel (default: 400)
- Additional layout parameters for advanced customization

## Using the GUI Interface

Once launched with `pf.gui`, the interface provides several key areas:

### 1. Workflow Graph Panel  
This interactive canvas is where you build and manage your workflow visually. Key features include:

- **Adding nodes**: Browse the Node Store (left panel), click any node to instantly add it to the canvas (appears to the right of existing workflow)
- **Creating connections**: Click and drag from **output ports** (right side of nodes) to **input ports** (left side of nodes)
- **Special connections**: For higher-order functions, connect to the **large circular port** in the upper-right corner of nodes (not regular input/output ports)
- **Deleting elements**: **Select any node or connection** and press the **Delete key** on your keyboard
- **Navigation controls**:
    - **Zoom**: Scroll with mouse wheel OR use **+/- buttons** in top toolbar
    - **Pan**: Click and drag **any empty area** of the canvas
    - **Minimap**: Toggle the **overview map** (bottom-left corner) for quick navigation in complex workflows
- **Selection**: Click nodes to select (highlighted with blue border) for configuration or deletion
- **Rename Node Label**: Select node and double click its node label to rename it

> 💡 **Pro Tip**: All standard canvas operations work intuitively - no special knowledge required. Just click, drag, and use your keyboard like in any modern visual editor.

### 2. Node Library Panel
- **Built-in Nodes**: Predefined computational nodes organized by category
- **Custom Nodes (planned)**: Your own node definitions (automatically detected)
- **Search Function (planned)**: Quickly find nodes by name or functionality

### 3. Output Panel
- View outputs of executed nodes 
- Includes print statements in node (e.g. for debugging)

### 4. Logging Info
- View log info and error messages (e.g. if node fails)

### 5. Node Buttons
- **Run**: Execute the workflow until that node
- **Source**: Show the source code of the node
- **Expand/Collapse**: Expand/Collapse workflow nodes

## Workflow Management

### Creating and Saving New Workflows
1. Click the "+" button in tab bar
2. Enter a name for your workflow (in upper-right text box and press enter)
3. Save node by clicking save icon in upper toolbar

Workflows are saved to the directory specified in `workflow_path` (default: system-dependent path).

## Node Operations

### Adding Nodes
1. Find a node in the Node Library panel
2. Drag it to the canvas
3. Position it where desired

### Connecting Nodes
1. Click and hold on an output port (right side of node)
2. Drag to an input port (left side of another node)
3. Release to create the connection

### Configuring Nodes
1. Click on a node to select it
2. Edit parameters in the Properties Panel
3. Changes are automatically saved

### Node Context Menu
Right-click any node for additional options:

- **Duplicate (planned)**: Create a copy of the node
- **Run**: Run just this node (and prerequisites)
- **Source**: See the node's implementation
- **Documentation (planned)**: View detailed node documentation

## Advanced Features (planned)

### Workflow Templates
Save frequently used workflow patterns as templates:
1. Create your workflow
2. Right-click the workflow tab
3. Select "Save as Template"
4. Name your template
5. Access templates via the "Templates" menu

### Batch Execution
Run multiple parameter sets automatically:
1. Right-click a node with configurable parameters
2. Select "Create Parameter Sweep"
3. Define parameter ranges
4. Execute to run all combinations

### Debugging Tools
- **Breakpoints**: Pause execution at specific nodes
- **Variable Inspection**: View intermediate values
- **Execution Timeline**: Visualize node execution order
- **Error Highlighting**: Identify problematic nodes

### Collaboration Features
- **Share Workflows**: Generate shareable links
- **Commenting**: Add notes to nodes/workflows
- **User Permissions**: Control access to workflows

### Version Control
The GUI integrates with workflow versioning:

- Automatic versioning on save
- Compare different versions
- Revert to previous versions
- Add version comments for documentation

## Practical Examples

### Example 1: Basic Workflow Creation

```python
import pyiron_core as pc

# Initialize with custom workflow path
pf = pc.PyironFlow(
    workflow_path="./my_workflows",
    hash_nodes=True
)

# Launch GUI
pf.gui
```

1. In the GUI:
   - Click "New Workflow" and name it "Data Analysis"
   - From the "Data" category, drag a "Load CSV" node to the canvas
   - From the "Processing" category, drag a "Normalize Data" node
   - Connect the output of "Load CSV" to the input of "Normalize Data"
   - Configure the "Load CSV" node with your file path
   - Click "Run" to execute the workflow

### Example 2: Parameter Sweep

```python
import pyiron_core as pc

pf = pc.PyironFlow()
pf.gui
```

1. Create a workflow with a node that has a configurable parameter (e.g., "amplitude")
2. Right-click the node and select "Create Parameter Sweep"
3. Define the parameter range (e.g., min=0.1, max=1.0, steps=10)
4. Execute the sweep
5. View results in the Results Panel as a table or plot



This setup allows:

- Multiple users to access the same workflows
- Persistent storage of workflow results
- Advanced search capabilities across workflow history
- Integration with other pyiron_core tools

## Best Practices

### Organizing Complex Workflows
- Use **sub-workflows** for modular design
- Group related nodes with **frames** (right-click canvas → "Add Frame")
- Use **comments** liberally to document your workflow
- Maintain consistent **naming conventions** for nodes

### Performance Tips
- Enable `hash_nodes=True` for large workflows
- Use **workflow templates** for common patterns
- Save frequently to avoid data loss
- For very large data, consider using **database storage** instead of file-based

### Collaboration Guidelines
- Add descriptive **workflow descriptions**
- Use **version comments** to document changes
- Share only finalized workflows (not work-in-progress)
- Use the **commenting system** for feedback

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| GUI doesn't launch | Check browser permissions; try a different browser |
| Nodes not appearing in library | Verify nodes are properly defined at module level |
| Connections not working | Ensure compatible data types between connected ports |
| Workflow execution fails | Check node logs in Results Panel for error details |
| Slow performance with large workflows | Enable `hash_nodes=True` in initialization |

## Conclusion

The PyironFlow GUI transforms scientific workflow development from a coding-intensive task into a visual, intuitive process. By combining the power of pyiron_core's workflow engine with an accessible graphical interface, researchers can focus on their scientific questions rather than implementation details.

Whether you're building simple data processing pipelines or complex multi-stage simulations, the PyironFlow GUI provides the tools you need to design, execute, and share reproducible scientific workflows with minimal coding effort.