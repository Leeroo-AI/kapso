# =============================================================================
# Cognitive System Report Generator
# =============================================================================
# Generates human-readable HTML reports of cognitive flow execution.
# =============================================================================

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ReportSection:
    """A section in the report."""
    title: str
    icon: str
    content: str
    status: str = "info"  # info, success, warning, error
    subsections: List["ReportSection"] = field(default_factory=list)


@dataclass 
class CognitiveReport:
    """Complete cognitive execution report."""
    goal: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    sections: List[ReportSection] = field(default_factory=list)
    final_score: Optional[float] = None
    total_iterations: int = 0
    total_cost: float = 0.0
    
    def add_section(self, section: ReportSection):
        self.sections.append(section)
    
    def to_html(self) -> str:
        """Generate HTML report."""
        duration = ""
        if self.ended_at and self.started_at:
            delta = self.ended_at - self.started_at
            duration = f"{delta.total_seconds():.1f}s"
        
        # Status badge
        if self.final_score is not None:
            if self.final_score >= 0.7:
                status_badge = '<span class="badge success">✅ PASSED</span>'
            else:
                status_badge = '<span class="badge error">❌ FAILED</span>'
        else:
            status_badge = '<span class="badge warning">⏳ IN PROGRESS</span>'
        
        sections_html = "\n".join(self._render_section(s) for s in self.sections)
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Execution Report</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2937;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #374151;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1000px; margin: 0 auto; }}
        
        header {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        
        h1 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--accent);
        }}
        
        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .meta-item {{
            background: var(--bg-primary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .meta-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        
        .meta-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        
        .badge.success {{ background: var(--success); color: white; }}
        .badge.warning {{ background: var(--warning); color: black; }}
        .badge.error {{ background: var(--error); color: white; }}
        
        .section {{
            background: var(--bg-card);
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            overflow: hidden;
        }}
        
        .section-header {{
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        
        .section-icon {{ font-size: 1.25rem; }}
        .section-title {{ font-weight: 600; }}
        
        .section-content {{
            padding: 1.5rem;
        }}
        
        .subsection {{
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid var(--accent);
        }}
        
        .subsection:last-child {{ margin-bottom: 0; }}
        
        .subsection-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }}
        
        pre {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.875rem;
            border: 1px solid var(--border);
        }}
        
        code {{
            font-family: 'SF Mono', 'Fira Code', monospace;
        }}
        
        ul {{ list-style: none; }}
        
        li {{
            padding: 0.25rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        li::before {{
            content: "→";
            position: absolute;
            left: 0;
            color: var(--accent);
        }}
        
        .step-grid {{
            display: grid;
            gap: 0.75rem;
        }}
        
        .step {{
            display: grid;
            grid-template-columns: 2rem 1fr;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg-primary);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .step-num {{
            background: var(--accent);
            color: white;
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .step-info {{ flex: 1; }}
        .step-title {{ font-weight: 600; }}
        .step-impl {{ font-size: 0.875rem; color: var(--text-secondary); }}
        .step-heuristics {{ font-size: 0.75rem; color: var(--success); }}
        
        .highlight {{ color: var(--success); }}
        .warning-text {{ color: var(--warning); }}
        .error-text {{ color: var(--error); }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Cognitive Execution Report</h1>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">{self.goal[:100]}...</p>
            {status_badge}
            
            <div class="meta">
                <div class="meta-item">
                    <div class="meta-label">Started</div>
                    <div class="meta-value">{self.started_at.strftime('%H:%M:%S')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Duration</div>
                    <div class="meta-value">{duration}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Iterations</div>
                    <div class="meta-value">{self.total_iterations}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Final Score</div>
                    <div class="meta-value">{f"{self.final_score:.0%}" if self.final_score else "N/A"}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Cost</div>
                    <div class="meta-value">${self.total_cost:.3f}</div>
                </div>
            </div>
        </header>
        
        {sections_html}
        
        <footer>
            Generated by Praxium Cognitive System
        </footer>
    </div>
</body>
</html>'''
    
    def _render_section(self, section: ReportSection) -> str:
        subsections_html = ""
        if section.subsections:
            subsections_html = "\n".join(
                f'''<div class="subsection">
                    <div class="subsection-title">{s.icon} {s.title}</div>
                    <div>{s.content}</div>
                </div>'''
                for s in section.subsections
            )
        
        return f'''
        <div class="section">
            <div class="section-header">
                <span class="section-icon">{section.icon}</span>
                <span class="section-title">{section.title}</span>
            </div>
            <div class="section-content">
                {section.content}
                {subsections_html}
            </div>
        </div>
        '''
    
    def save(self, path: Path):
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html())


class CognitiveReportBuilder:
    """Builder for creating cognitive reports during execution."""
    
    def __init__(self, goal: str):
        self.report = CognitiveReport(
            goal=goal,
            started_at=datetime.now(),
        )
    
    def add_goal_initialization(
        self,
        goal_type: str,
        workflow_title: Optional[str],
        workflow_confidence: float,
        steps: List[Dict[str, Any]],
    ):
        """Add goal initialization section."""
        if workflow_title:
            # Count total heuristics from all steps
            total_heuristics = sum(step.get('heuristics', 0) for step in steps)
            
            # Build steps HTML
            steps_html = '<div class="step-grid">'
            for step in steps:
                impl = step.get('implementation', 'N/A')
                heuristics = step.get('heuristics', 0)
                steps_html += f'''
                <div class="step">
                    <div class="step-num">{step['number']}</div>
                    <div class="step-info">
                        <div class="step-title">{step['title']}</div>
                        <div class="step-impl">→ {impl}</div>
                        <div class="step-heuristics">{heuristics} heuristics</div>
                    </div>
                </div>
                '''
            steps_html += '</div>'
            
            content = f'''
            <p><strong>Goal Type:</strong> {goal_type}</p>
            <p><strong>Workflow:</strong> {workflow_title}</p>
            <p><strong>Confidence:</strong> <span class="highlight">{workflow_confidence:.0%}</span></p>
            <p><strong>Total Heuristics:</strong> {total_heuristics} (from graph)</p>
            <h4 style="margin: 1rem 0 0.5rem;">Workflow Steps:</h4>
            {steps_html}
            '''
        else:
            content = f'''
            <p><strong>Goal Type:</strong> {goal_type}</p>
            <p class="warning-text">No workflow match found - using synthesized plan</p>
            '''
        
        self.report.add_section(ReportSection(
            title="Goal Initialization",
            icon="🎯",
            content=content,
        ))
    
    def add_briefing(
        self,
        iteration: int,
        current_step: Dict[str, Any],
        episodic_status: str,
        context_sizes: Dict[str, int],
    ):
        """Add briefing section."""
        heuristics_html = ""
        if current_step.get('heuristics_content'):
            heuristics_html = "<ul>"
            for h in current_step['heuristics_content'][:3]:
                heuristics_html += f"<li>{h[:80]}...</li>"
            heuristics_html += "</ul>"
        
        content = f'''
        <p><strong>Iteration:</strong> {iteration}</p>
        <p><strong>Current Step:</strong> {current_step.get('number', '?')}/{current_step.get('total', '?')} - {current_step.get('title', 'Unknown')}</p>
        <p><strong>Implementation:</strong> {current_step.get('implementation', 'N/A')}</p>
        <p><strong>Episodic Memory:</strong> {episodic_status}</p>
        
        <h4 style="margin: 1rem 0 0.5rem;">Heuristics:</h4>
        {heuristics_html}
        
        <h4 style="margin: 1rem 0 0.5rem;">Context Sent to Agent:</h4>
        <ul>
            <li>Problem: {context_sizes.get('problem', 0):,} chars</li>
            <li>Workflow guidance: {context_sizes.get('workflow', 0):,} chars</li>
            <li>Implementation code: {context_sizes.get('code', 0):,} chars</li>
        </ul>
        '''
        
        self.report.add_section(ReportSection(
            title=f"Briefing (Iteration {iteration})",
            icon="📋",
            content=content,
        ))
    
    def add_execution(
        self,
        iteration: int,
        agent: str,
        duration: float,
        cost: float,
        output: str,
    ):
        """Add execution section."""
        content = f'''
        <p><strong>Agent:</strong> {agent}</p>
        <p><strong>Duration:</strong> {duration:.1f}s</p>
        <p><strong>Cost:</strong> ${cost:.4f}</p>
        
        <h4 style="margin: 1rem 0 0.5rem;">Output:</h4>
        <pre><code>{output[:500]}{'...' if len(output) > 500 else ''}</code></pre>
        '''
        
        self.report.add_section(ReportSection(
            title=f"Agent Execution (Iteration {iteration})",
            icon="🤖",
            content=content,
        ))
    
    def add_evaluation(
        self,
        iteration: int,
        score: float,
        feedback: str,
        decision: Optional[str] = None,
    ):
        """Add evaluation section."""
        score_class = "highlight" if score >= 0.7 else "warning-text" if score >= 0.5 else "error-text"
        
        decision_html = ""
        if decision:
            decision_html = f'<p><strong>Decision:</strong> {decision}</p>'
        
        content = f'''
        <p><strong>Score:</strong> <span class="{score_class}">{score:.0%}</span></p>
        {decision_html}
        
        <h4 style="margin: 1rem 0 0.5rem;">Feedback:</h4>
        <p>{feedback}</p>
        '''
        
        self.report.add_section(ReportSection(
            title=f"Evaluation (Iteration {iteration})",
            icon="📊",
            content=content,
        ))
    
    def finalize(
        self,
        final_score: float,
        total_iterations: int,
        total_cost: float,
    ) -> CognitiveReport:
        """Finalize and return the report."""
        self.report.ended_at = datetime.now()
        self.report.final_score = final_score
        self.report.total_iterations = total_iterations
        self.report.total_cost = total_cost
        return self.report

