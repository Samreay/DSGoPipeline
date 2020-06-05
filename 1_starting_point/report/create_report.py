import subprocess
import os
import inspect

here = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
filename = os.path.join(here, "weekly_report.ipynb")
output_dir = os.path.join(here, "output")
print(f"Outputting report to {output_dir}")
os.makedirs(output_dir, exist_ok=True)
command = f"jupyter nbconvert --no-input --to html --ExecutePreprocessor.store_widget_state=True --execute  --output-dir {output_dir} {filename}"
subprocess.run(command, shell=True)
