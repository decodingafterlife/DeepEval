
import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, render_template
from run_evaluation import run_evaluation_main
import config

app = Flask(__name__)

tasks = {}

def run_background_task(task_id, eval_type):
    all_results = None
    try:
        tasks[task_id]['status'] = f'running_{eval_type}_evaluation'
        report_content, report_filename = run_evaluation_main(eval_type)
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = {"report_content": report_content}

    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run-evaluation/<eval_type>', methods=['POST'])
def start_evaluation(eval_type):
    if eval_type not in ['quick', 'deep']:
        return jsonify({"error": "Invalid evaluation type"}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'starting'}
    
    thread = threading.Thread(target=run_background_task, args=(task_id, eval_type))
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

if __name__ == '__main__':
    app.run(debug=True)
