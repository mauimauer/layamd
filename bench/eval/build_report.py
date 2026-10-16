"""Embed report_data.json into report_template.html -> report.html (self-contained)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
data = json.dumps(json.load(open(os.path.join(HERE, "report_data.json"))), separators=(",", ":")).replace("</", "<\\/")
html = open(os.path.join(HERE, "report_template.html")).read().replace("/*DATA*/", data, 1)
open(os.path.join(HERE, "report.html"), "w").write(html)
print("report.html", len(html) // 1024, "KB")
