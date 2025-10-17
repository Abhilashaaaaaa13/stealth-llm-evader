import random
import requests
from pathlib import Path
import csv
import logging

logger = logging.getLogger(__name__)

def run_evaluation(files: list[Path], detectors_config: dict)->dict:
    """Eval detection scores. Mock for now; ad real API"""
    results = {'avg_detection':0, 'scores':[]}

    for file in files:
        text = file.read_text()
        #mock score: random <20% for demo
        score = random.uniform(0.05, 0.18)

        #real example(uncomment with api key )
        #response = requests.post(detectors_config['zerogpt_ur'l], json={'text':text})
        # score = response.json()['ai_probability']
        results['scores'].append({'file':file.name, 'detection':score})

    results['avg_detection'] = sum(s['detection'] for s in results['scores']) / len(results['scores'])
    #save report
    report_path = Path('output/eval_reports/report.csv')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path,'w',newlines='') as f:
        writer = csv.DictWriter(f, fieldnames=['file','detection'])
        writer.writeheader()
        writer.writerows(results['scores'])
    
    return results

