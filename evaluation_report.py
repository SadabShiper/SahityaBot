import pandas as pd
import json
from datetime import datetime

class RAGEvaluator:
    def __init__(self, log_file="evaluation_log.jsonl"):
        self.log_file = log_file
    
    def generate_report(self, output_format="markdown"):
        """Generate evaluation report in specified format"""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f.readlines()]
            
            if not logs:
                return "No evaluation data available"
                
            df = pd.DataFrame(logs)
            
            # Calculate metrics
            metrics = {
                "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_interactions": len(df),
                "average_groundedness": round(df["groundedness"].mean(), 3),
                "average_relevance": round(df["relevance"].mean(), 3),
                "average_combined": round(df["combined_score"].mean(), 3),
                "groundedness_distribution": self._get_score_distribution(df["groundedness"]),
                "relevance_distribution": self._get_score_distribution(df["relevance"]),
                "top_questions": self._get_top_questions(df),
                "bottom_questions": self._get_bottom_questions(df)
            }
            
            if output_format == "markdown":
                return self._generate_markdown_report(metrics, df)
            elif output_format == "html":
                return self._generate_html_report(metrics, df)
            else:
                return metrics
                
        except FileNotFoundError:
            return "Evaluation log file not found"
    
    def _get_score_distribution(self, series):
        return {
            "0.0-0.2": len(series[series <= 0.2]),
            "0.2-0.4": len(series[(series > 0.2) & (series <= 0.4)]),
            "0.4-0.6": len(series[(series > 0.4) & (series <= 0.6)]),
            "0.6-0.8": len(series[(series > 0.6) & (series <= 0.8)]),
            "0.8-1.0": len(series[series > 0.8])
        }
    
    def _get_top_questions(self, df, n=3):
        return (df.sort_values("combined_score", ascending=False)
                .head(n)[["question", "answer", "combined_score"]]
                .to_dict("records"))
    
    def _get_bottom_questions(self, df, n=3):
        return (df.sort_values("combined_score")
                .head(n)[["question", "answer", "combined_score"]]
                .to_dict("records"))
    
    def _generate_markdown_report(self, metrics, df):
        """Generate markdown formatted report"""
        report = f"""
# RAG System Evaluation Report
**Generated on**: {metrics['report_date']}  
**Total Interactions**: {metrics['total_interactions']}

## Overall Metrics
| Metric | Value |
|--------|-------|
| Average Groundedness | {metrics['average_groundedness']} |
| Average Relevance | {metrics['average_relevance']} |
| Combined Score | {metrics['average_combined']} |

## Score Distributions
### Groundedness
{self._distribution_to_markdown(metrics['groundedness_distribution'])}

### Relevance
{self._distribution_to_markdown(metrics['relevance_distribution'])}

## Top Performing Questions
{self._questions_to_markdown(metrics['top_questions'])}

## Questions Needing Improvement
{self._questions_to_markdown(metrics['bottom_questions'])}

## Full Evaluation Data
{df.to_markdown(index=False)}
"""
        return report
    
    def _distribution_to_markdown(self, dist):
        return "\n".join([f"- {k}: {v} interactions" for k, v in dist.items()])
    
    def _questions_to_markdown(self, questions):
        return "\n".join([
            f"1. **Question**: {q['question']}\n   **Answer**: {q['answer']}\n   **Score**: {q['combined_score']:.3f}"
            for q in questions
        ])
    
    def _generate_html_report(self, metrics, df):
        """Generate HTML formatted report"""
        html = f"""
<html>
<head>
<title>RAG Evaluation Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2, h3 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    .good {{ background-color: #e6ffe6; }}
    .bad {{ background-color: #ffe6e6; }}
    .metric {{ font-weight: bold; }}
</style>
</head>
<body>
<h1>RAG System Evaluation Report</h1>
<p><span class="metric">Generated on:</span> {metrics['report_date']}</p>
<p><span class="metric">Total Interactions:</span> {metrics['total_interactions']}</p>

<h2>Overall Metrics</h2>
<table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td>Average Groundedness</td>
        <td>{metrics['average_groundedness']}</td>
    </tr>
    <tr>
        <td>Average Relevance</td>
        <td>{metrics['average_relevance']}</td>
    </tr>
    <tr>
        <td>Combined Score</td>
        <td>{metrics['average_combined']}</td>
    </tr>
</table>

<h2>Score Distributions</h2>
<h3>Groundedness</h3>
{self._distribution_to_html(metrics['groundedness_distribution'])}

<h3>Relevance</h3>
{self._distribution_to_html(metrics['relevance_distribution'])}

<h2>Top Performing Questions</h2>
{self._questions_to_html(metrics['top_questions'], "good")}

<h2>Questions Needing Improvement</h2>
{self._questions_to_html(metrics['bottom_questions'], "bad")}

<h2>Full Evaluation Data</h2>
{df.to_html(index=False)}
</body>
</html>
"""
        return html
    
    def _distribution_to_html(self, dist):
        items = "".join([f"<li>{k}: {v} interactions</li>" for k, v in dist.items()])
        return f"<ul>{items}</ul>"
    
    def _questions_to_html(self, questions, css_class):
        items = []
        for i, q in enumerate(questions, 1):
            items.append(f"""
<div class="{css_class}">
<h3>{i}. Question</h3>
<p>{q['question']}</p>
<h4>Answer</h4>
<p>{q['answer']}</p>
<p><strong>Score:</strong> {q['combined_score']:.3f}</p>
</div>
""")
        return "\n".join(items)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Generate and save reports
    markdown_report = evaluator.generate_report("markdown")
    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    html_report = evaluator.generate_report("html")
    with open("evaluation_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print("Evaluation reports generated:")
    print("- evaluation_report.md")
    print("- evaluation_report.html")