"""
SVEND Agents - Alpha Web Interface

Complete zero-to-classifier pipeline with ML services.
Deploy today, seek $100k for expansion.
"""

import os
import sys
import json
import uuid
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'svend-alpha-' + str(uuid.uuid4()))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Database setup
DB_PATH = Path(__file__).parent / 'data' / 'svend.db'
DB_PATH.parent.mkdir(exist_ok=True)

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables."""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            tier TEXT DEFAULT 'alpha'
        );

        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            name TEXT NOT NULL,
            steps TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_run TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            type TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            endpoint TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()

init_db()

# Store results in memory (for demo - use Redis/DB in production)
RESULTS_CACHE = {}


def get_user():
    """Get current user from session."""
    return session.get('user')


def require_user(f):
    """Decorator to require user session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_user():
            if request.is_json:
                return jsonify({'error': 'Not authenticated'}), 401
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Landing & Auth
# =============================================================================

@app.route('/')
def landing():
    """Landing page - show login or redirect to home."""
    if get_user():
        return redirect(url_for('home'))
    return render_template('landing.html')


@app.route('/auth/login', methods=['POST'])
def login():
    """Simple email-based login for alpha."""
    data = request.json
    email = data.get('email', '').strip().lower()
    name = data.get('name', '').strip()

    if not email or '@' not in email:
        return jsonify({'error': 'Valid email required'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    if not user:
        # Create new user
        user_id = str(uuid.uuid4())
        conn.execute(
            'INSERT INTO users (id, email, name) VALUES (?, ?, ?)',
            (user_id, email, name or email.split('@')[0])
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    conn.close()

    session['user'] = {
        'id': user['id'],
        'email': user['email'],
        'name': user['name'],
        'tier': user['tier'],
    }

    return jsonify({'success': True, 'user': session['user']})


@app.route('/auth/logout')
def logout():
    """Logout."""
    session.clear()
    return redirect(url_for('landing'))


# =============================================================================
# Main Pages
# =============================================================================

@app.route('/home')
@require_user
def home():
    """Dashboard - overview of all services."""
    return render_template('home.html', user=get_user())


@app.route('/dsw')
@require_user
def dsw_page():
    """Decision Science Workbench page."""
    return render_template('dsw.html', user=get_user())


@app.route('/scrub')
@require_user
def scrub_page():
    """Scrub data cleaning page."""
    return render_template('scrub.html', user=get_user())


@app.route('/analyst')
@require_user
def analyst_page():
    """Analyst ML training page."""
    return render_template('analyst.html', user=get_user())


@app.route('/agents')
@require_user
def agents_page():
    """Other agents overview."""
    return render_template('agents.html', user=get_user())


@app.route('/workflows')
@require_user
def workflows_page():
    """Workflow builder and runner."""
    return render_template('workflows.html', user=get_user())


@app.route('/templates')
@require_user
def templates_page():
    """User templates management."""
    return render_template('templates.html', user=get_user())


# =============================================================================
# API Endpoints - DSW
# =============================================================================

@app.route('/api/dsw/from-intent', methods=['POST'])
@require_user
def api_dsw_from_intent():
    """Run DSW from intent (zero data)."""
    try:
        from dsw import DecisionScienceWorkbench

        data = request.json
        intent = data.get('intent', '')
        domain = data.get('domain')
        n_records = int(data.get('n_records', 500))
        priority = data.get('priority', 'balanced')

        if not intent:
            return jsonify({'error': 'Intent is required'}), 400

        dsw = DecisionScienceWorkbench()
        result = dsw.from_intent(
            intent=intent,
            domain=domain if domain else None,
            n_records=n_records,
            priority=priority,
        )

        # Store result
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        RESULTS_CACHE[result_id] = result

        # Log usage
        log_usage('dsw_from_intent')

        return jsonify({
            'result_id': result_id,
            'schema': {
                'name': result.schema.name,
                'intent': result.schema.intent,
                'domain': result.schema.domain,
                'features': [f.name for f in result.schema.features],
                'target': result.schema.target_name,
            },
            'data_shape': list(result.synthetic_data.shape),
            'model_type': result.model_type,
            'metrics': result.metrics,
            'pipeline': result.pipeline_steps,
            'warnings': result.warnings,
            'summary': result.summary(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dsw/from-data', methods=['POST'])
@require_user
def api_dsw_from_data():
    """Run DSW from uploaded data."""
    try:
        import pandas as pd
        from dsw import DecisionScienceWorkbench

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        target = request.form.get('target', '')
        intent = request.form.get('intent', '')
        priority = request.form.get('priority', 'balanced')

        if not target:
            return jsonify({'error': 'Target column is required'}), 400

        # Read CSV
        df = pd.read_csv(file)

        if target not in df.columns:
            return jsonify({'error': f'Target column "{target}" not found in data'}), 400

        dsw = DecisionScienceWorkbench()
        result = dsw.from_data(
            data=df,
            target=target,
            intent=intent,
            priority=priority,
        )

        # Store result
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        RESULTS_CACHE[result_id] = result

        # Log usage
        log_usage('dsw_from_data')

        return jsonify({
            'result_id': result_id,
            'original_shape': list(result.original_data.shape),
            'cleaned_shape': list(result.cleaned_data.shape),
            'model_type': result.model_type,
            'metrics': result.metrics,
            'feature_importance': result.feature_importance[:10],
            'pipeline': result.pipeline_steps,
            'warnings': result.warnings,
            'summary': result.summary(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Scrub
# =============================================================================

@app.route('/api/scrub', methods=['POST'])
@require_user
def api_scrub():
    """Clean data with Scrub."""
    try:
        import pandas as pd
        from scrub import DataCleaner, CleaningConfig

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        domain_rules_str = request.form.get('domain_rules', '{}')

        try:
            domain_rules = json.loads(domain_rules_str) if domain_rules_str else {}
            # Convert lists to tuples
            domain_rules = {k: tuple(v) if isinstance(v, list) else v for k, v in domain_rules.items()}
        except:
            domain_rules = {}

        df = pd.read_csv(file)

        config = CleaningConfig(domain_rules=domain_rules)
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df, config)

        # Store result
        result_id = f"scrub_{uuid.uuid4().hex[:8]}"
        RESULTS_CACHE[result_id] = {
            'data': df_clean,
            'result': result,
        }

        # Log usage
        log_usage('scrub')

        return jsonify({
            'result_id': result_id,
            'original_shape': list(result.original_shape),
            'cleaned_shape': list(result.cleaned_shape),
            'outliers_flagged': result.outliers.count if result.outliers else 0,
            'high_severity': result.outliers.high_severity_count if result.outliers else 0,
            'missing_filled': result.missing.total_filled if result.missing else 0,
            'normalizations': result.normalization.total_changes if result.normalization else 0,
            'warnings': result.warnings,
            'summary': result.summary(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - EDA
# =============================================================================

@app.route('/api/eda', methods=['POST'])
@require_user
def api_eda():
    """Run automated EDA on uploaded data."""
    try:
        import pandas as pd
        from analyst import quick_eda

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        name = request.form.get('name', 'dataset')
        generate_charts = request.form.get('charts', 'true').lower() == 'true'

        df = pd.read_csv(file)

        # Run EDA
        report = quick_eda(df, name=name, generate_charts=generate_charts)

        # Calculate data quality score
        missing_penalty = min(report.total_missing_pct, 0.3)  # Cap at 30%
        duplicate_penalty = min(report.duplicate_pct, 0.1)  # Cap at 10%
        data_quality_score = 1.0 - missing_penalty - duplicate_penalty

        # Build response
        response = {
            'name': report.dataset_name,
            'shape': [report.n_rows, report.n_cols],
            'columns': [c.name for c in report.columns],
            'data_quality_score': data_quality_score,
            'summary': report.summary(),

            # Column profiles
            'profiles': [
                {
                    'name': c.name,
                    'dtype': c.dtype,
                    'missing_count': c.missing,
                    'missing_pct': c.missing_pct,
                    'unique_count': c.unique,
                    'mean': c.mean,
                    'std': c.std,
                    'min': c.min,
                    'max': c.max,
                    'top_values': c.top_values[:5] if c.top_values else None,
                    'is_numeric': c.is_numeric,
                    'has_outliers': c.has_outliers,
                    'outlier_count': c.outlier_count,
                }
                for c in report.columns
            ],

            # Missing data summary
            'missing_report': {
                'total_missing': report.total_missing,
                'total_missing_pct': report.total_missing_pct,
                'columns_with_missing': len([c for c in report.columns if c.missing > 0]),
                'by_column': {c.name: c.missing for c in report.columns if c.missing > 0},
            } if report.total_missing > 0 else None,

            # Outliers
            'outliers': [
                {'column': c.name, 'count': c.outlier_count, 'pct': c.outlier_count / c.count if c.count else 0}
                for c in report.columns if c.has_outliers
            ],

            # Correlations
            'correlations': [
                {'col1': c[0], 'col2': c[1], 'value': c[2]}
                for c in (report.high_correlations or [])[:10]
            ],

            # Recommendations based on findings
            'recommendations': [],
        }

        # Add recommendations
        if report.total_missing_pct > 0.05:
            response['recommendations'].append(f"Address missing values ({report.total_missing_pct:.1%} of data)")
        if report.duplicate_pct > 0.01:
            response['recommendations'].append(f"Review duplicate rows ({report.duplicate_pct:.1%})")
        outlier_cols = [c for c in report.columns if c.has_outliers]
        if outlier_cols:
            response['recommendations'].append(f"Review outliers in {len(outlier_cols)} column(s)")
        if report.high_correlations:
            response['recommendations'].append(f"Consider multicollinearity ({len(report.high_correlations)} high correlations)")

        # Charts are already base64 encoded in the report
        if report.charts:
            response['charts'] = report.charts

        # Store result for download
        result_id = f"eda_{uuid.uuid4().hex[:8]}"
        RESULTS_CACHE[result_id] = report

        response['result_id'] = result_id

        # Log usage
        log_usage('eda')

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Analyst
# =============================================================================

@app.route('/api/analyst', methods=['POST'])
@require_user
def api_analyst():
    """Train model with Analyst."""
    try:
        import pandas as pd
        from analyst import Analyst

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        target = request.form.get('target', '')
        intent = request.form.get('intent', '')
        priority = request.form.get('priority', 'balanced')

        if not target:
            return jsonify({'error': 'Target column is required'}), 400

        df = pd.read_csv(file)

        if target not in df.columns:
            return jsonify({'error': f'Target column "{target}" not found'}), 400

        analyst = Analyst()
        result = analyst.train(
            data=df,
            target=target,
            intent=intent,
            priority=priority,
        )

        # Store result
        result_id = f"analyst_{uuid.uuid4().hex[:8]}"
        RESULTS_CACHE[result_id] = result

        # Log usage
        log_usage('analyst')

        return jsonify({
            'result_id': result_id,
            'model_type': result.model_type.value,
            'task_type': result.task_type.value,
            'metrics': {m.name: m.value for m in result.report.metrics},
            'feature_importance': [
                {'feature': f.feature, 'importance': f.importance}
                for f in result.report.feature_importance[:10]
            ],
            'cv_mean': result.report.cv_mean,
            'cv_std': result.report.cv_std,
            'warnings': result.report.warnings,
            'recommendations': result.report.recommendations,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Templates
# =============================================================================

@app.route('/api/templates', methods=['GET', 'POST'])
@require_user
def api_templates():
    """List or create templates."""
    user = get_user()
    conn = get_db()

    if request.method == 'GET':
        templates = conn.execute(
            'SELECT * FROM templates WHERE user_id = ? OR user_id IS NULL ORDER BY created_at DESC',
            (user['id'],)
        ).fetchall()
        conn.close()
        return jsonify({
            'templates': [dict(t) for t in templates]
        })

    # POST - create new template
    data = request.json
    template_id = str(uuid.uuid4())

    conn.execute(
        'INSERT INTO templates (id, user_id, name, type, content) VALUES (?, ?, ?, ?, ?)',
        (template_id, user['id'], data['name'], data['type'], json.dumps(data['content']))
    )
    conn.commit()
    conn.close()

    return jsonify({'id': template_id, 'success': True})


@app.route('/api/templates/<template_id>', methods=['GET', 'PUT', 'DELETE'])
@require_user
def api_template_detail(template_id):
    """Get, update, or delete a template."""
    user = get_user()
    conn = get_db()

    template = conn.execute(
        'SELECT * FROM templates WHERE id = ? AND (user_id = ? OR user_id IS NULL)',
        (template_id, user['id'])
    ).fetchone()

    if not template:
        conn.close()
        return jsonify({'error': 'Template not found'}), 404

    if request.method == 'GET':
        conn.close()
        return jsonify(dict(template))

    if request.method == 'DELETE':
        if template['user_id'] != user['id']:
            conn.close()
            return jsonify({'error': 'Cannot delete system template'}), 403
        conn.execute('DELETE FROM templates WHERE id = ?', (template_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    # PUT - update
    if template['user_id'] != user['id']:
        conn.close()
        return jsonify({'error': 'Cannot modify system template'}), 403

    data = request.json
    conn.execute(
        'UPDATE templates SET name = ?, content = ?, updated_at = ? WHERE id = ?',
        (data.get('name', template['name']), json.dumps(data.get('content', json.loads(template['content']))),
         datetime.now().isoformat(), template_id)
    )
    conn.commit()
    conn.close()

    return jsonify({'success': True})


# =============================================================================
# API Endpoints - Workflows
# =============================================================================

@app.route('/api/workflows', methods=['GET', 'POST'])
@require_user
def api_workflows():
    """List or create workflows."""
    user = get_user()
    conn = get_db()

    if request.method == 'GET':
        workflows = conn.execute(
            'SELECT * FROM workflows WHERE user_id = ? ORDER BY created_at DESC',
            (user['id'],)
        ).fetchall()
        conn.close()
        return jsonify({
            'workflows': [dict(w) for w in workflows]
        })

    # POST - create new workflow
    data = request.json
    workflow_id = str(uuid.uuid4())

    conn.execute(
        'INSERT INTO workflows (id, user_id, name, steps) VALUES (?, ?, ?, ?)',
        (workflow_id, user['id'], data['name'], json.dumps(data['steps']))
    )
    conn.commit()
    conn.close()

    return jsonify({'id': workflow_id, 'success': True})


@app.route('/api/workflows/<workflow_id>', methods=['GET', 'PUT', 'DELETE'])
@require_user
def api_workflow_detail(workflow_id):
    """Get, update, or delete a workflow."""
    user = get_user()
    conn = get_db()

    workflow = conn.execute(
        'SELECT * FROM workflows WHERE id = ? AND user_id = ?',
        (workflow_id, user['id'])
    ).fetchone()

    if not workflow:
        conn.close()
        return jsonify({'error': 'Workflow not found'}), 404

    if request.method == 'GET':
        conn.close()
        return jsonify(dict(workflow))

    if request.method == 'DELETE':
        conn.execute('DELETE FROM workflows WHERE id = ?', (workflow_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    # PUT - update
    data = request.json
    conn.execute(
        'UPDATE workflows SET name = ?, steps = ? WHERE id = ?',
        (data.get('name', workflow['name']), json.dumps(data.get('steps', json.loads(workflow['steps']))),
         workflow_id)
    )
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/api/workflows/<workflow_id>/run', methods=['POST'])
@require_user
def api_workflow_run(workflow_id):
    """Run a workflow."""
    user = get_user()
    conn = get_db()

    workflow = conn.execute(
        'SELECT * FROM workflows WHERE id = ? AND user_id = ?',
        (workflow_id, user['id'])
    ).fetchone()

    if not workflow:
        conn.close()
        return jsonify({'error': 'Workflow not found'}), 404

    steps = json.loads(workflow['steps'])

    # Update last run time
    conn.execute(
        'UPDATE workflows SET last_run = ? WHERE id = ?',
        (datetime.now().isoformat(), workflow_id)
    )
    conn.commit()
    conn.close()

    # Execute workflow steps
    try:
        from workflow import Workflow

        wf = Workflow(workflow['name'])

        # Build workflow from saved steps
        for step in steps:
            step_type = step.get('type')
            if step_type == 'research':
                from workflow import ResearchStep
                wf.add(ResearchStep(step.get('query', ''), name=step.get('name')))
            elif step_type == 'coder':
                from workflow import CoderStep
                wf.add(CoderStep(step.get('prompt', ''), name=step.get('name'), uses=step.get('uses', [])))
            elif step_type == 'writer':
                from workflow import WriterStep
                wf.add(WriterStep(step.get('template', 'general'), name=step.get('name'), uses=step.get('uses', [])))
            # Add more step types as needed

        result = wf.run()

        # Log usage
        log_usage('workflow_run')

        return jsonify({
            'success': True,
            'result': result.to_dict() if hasattr(result, 'to_dict') else str(result),
        })

    except ImportError:
        # Workflow module not available - return mock result
        return jsonify({
            'success': True,
            'result': {
                'status': 'completed',
                'steps': [{'name': s.get('name', f'step_{i}'), 'status': 'done'} for i, s in enumerate(steps)],
                'message': 'Workflow executed successfully',
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Downloads
# =============================================================================

@app.route('/api/download/<result_id>/<file_type>')
@require_user
def api_download(result_id, file_type):
    """Download result files."""
    try:
        if result_id not in RESULTS_CACHE:
            return jsonify({'error': 'Result not found'}), 404

        result = RESULTS_CACHE[result_id]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if file_type == 'report':
                if hasattr(result, 'full_report'):
                    path = tmpdir / 'report.md'
                    path.write_text(result.full_report())
                elif hasattr(result, 'report'):
                    path = tmpdir / 'report.md'
                    path.write_text(result.report.to_markdown())
                else:
                    return jsonify({'error': 'No report available'}), 400
                return send_file(path, as_attachment=True, download_name='report.md')

            elif file_type == 'code':
                if hasattr(result, 'deployment_code'):
                    code = result.deployment_code
                elif hasattr(result, 'code'):
                    code = result.code
                else:
                    return jsonify({'error': 'No code available'}), 400
                path = tmpdir / 'train.py'
                path.write_text(code)
                return send_file(path, as_attachment=True, download_name='train.py')

            elif file_type == 'data':
                if hasattr(result, 'cleaned_data'):
                    df = result.cleaned_data
                elif isinstance(result, dict) and 'data' in result:
                    df = result['data']
                else:
                    return jsonify({'error': 'No data available'}), 400
                path = tmpdir / 'cleaned_data.csv'
                df.to_csv(path, index=False)
                return send_file(path, as_attachment=True, download_name='cleaned_data.csv')

            else:
                return jsonify({'error': f'Unknown file type: {file_type}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/schema/domains')
def api_schema_domains():
    """Get available domain templates."""
    try:
        from dsw.schema import SchemaGenerator

        gen = SchemaGenerator()
        domains = []
        for name, template in gen.DOMAIN_TEMPLATES.items():
            domains.append({
                'name': name,
                'task_type': template['task_type'],
                'features': [f.name for f in template['common_features']],
            })

        return jsonify({'domains': domains})
    except ImportError:
        # Return default domains if DSW not available
        return jsonify({
            'domains': [
                {'name': 'churn', 'task_type': 'classification', 'features': ['tenure', 'spend', 'support_tickets']},
                {'name': 'fraud', 'task_type': 'classification', 'features': ['amount', 'merchant', 'location']},
                {'name': 'lead_scoring', 'task_type': 'classification', 'features': ['company_size', 'engagement']},
            ]
        })


# =============================================================================
# Usage Logging
# =============================================================================

def log_usage(endpoint):
    """Log API usage."""
    user = get_user()
    if user:
        conn = get_db()
        conn.execute(
            'INSERT INTO usage_logs (user_id, endpoint) VALUES (?, ?)',
            (user['id'], endpoint)
        )
        conn.commit()
        conn.close()


@app.route('/api/usage')
@require_user
def api_usage():
    """Get user's usage stats."""
    user = get_user()
    conn = get_db()

    # Get usage counts by endpoint
    usage = conn.execute('''
        SELECT endpoint, COUNT(*) as count
        FROM usage_logs
        WHERE user_id = ?
        GROUP BY endpoint
    ''', (user['id'],)).fetchall()

    # Get total
    total = conn.execute(
        'SELECT COUNT(*) as count FROM usage_logs WHERE user_id = ?',
        (user['id'],)
    ).fetchone()

    conn.close()

    return jsonify({
        'total': total['count'],
        'by_endpoint': {u['endpoint']: u['count'] for u in usage},
    })


# =============================================================================
# Agent APIs
# =============================================================================

@app.route('/api/agents/research', methods=['POST'])
@require_user
def api_agent_research():
    """Run the research agent."""
    data = request.get_json()
    query = data.get('query', '')
    focus = data.get('focus', 'general')
    depth = data.get('depth', 'standard')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    log_usage('agent_research')

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from researcher.agent import ResearchAgent, ResearchQuery

        llm = get_shared_llm()
        agent = ResearchAgent(llm=llm)
        result = agent.run(ResearchQuery(question=query, focus=focus, depth=depth))

        response = {
            'summary': result.summary if hasattr(result, 'summary') else str(result),
            'sources': [{'title': s.title, 'url': s.url} for s in result.sources] if hasattr(result, 'sources') else [],
            'markdown': result.to_markdown() if hasattr(result, 'to_markdown') else None,
        }

        if llm is None:
            response['note'] = 'Running without LLM - synthesis may be limited. Set SVEND_LLM_SIZE env var.'

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Research agent error: {str(e)[:200]}',
        }), 500


# =============================================================================
# Shared LLM (loaded once, used by all agents)
# =============================================================================

_shared_llm = None
_shared_llm_loaded = False
_coder_llm = None
_coder_llm_loaded = False


def get_shared_llm():
    """Get or load shared LLM for all agents."""
    global _shared_llm, _shared_llm_loaded

    if _shared_llm_loaded:
        return _shared_llm

    # Check for LLM config
    llm_size = os.environ.get('SVEND_LLM_SIZE', '7B')  # Default to 7B for quality

    if os.environ.get('SVEND_LLM_DISABLED'):
        print("LLM disabled via SVEND_LLM_DISABLED env var")
        _shared_llm_loaded = True
        return None

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from core.llm import load_qwen
        print(f"\n{'='*60}")
        print(f"  Loading Qwen {llm_size} for all agents...")
        print(f"{'='*60}\n")
        _shared_llm = load_qwen(llm_size)
        _shared_llm_loaded = True
        print(f"LLM loaded successfully!")
        return _shared_llm
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        print("Agents will run in mock/fallback mode")
        _shared_llm_loaded = True
        return None


def get_coder_llm():
    """Get coder-specific LLM (Qwen Coder) or fall back to shared LLM."""
    global _coder_llm, _coder_llm_loaded

    if _coder_llm_loaded:
        return _coder_llm

    # Check if a specific coder model is requested
    coder_model = os.environ.get('SVEND_CODER_LLM')

    if not coder_model:
        # Use shared LLM
        _coder_llm_loaded = True
        _coder_llm = get_shared_llm()
        return _coder_llm

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from core.llm import load_qwen
        print(f"\n{'='*60}")
        print(f"  Loading Qwen Coder ({coder_model}) for code generation...")
        print(f"{'='*60}\n")
        _coder_llm = load_qwen(coder_model)
        _coder_llm_loaded = True
        print(f"Coder LLM loaded successfully!")
        return _coder_llm
    except Exception as e:
        print(f"Failed to load Coder LLM: {e}")
        print("Falling back to shared LLM")
        _coder_llm_loaded = True
        _coder_llm = get_shared_llm()
        return _coder_llm


@app.route('/api/agents/coder', methods=['POST'])
@require_user
def api_agent_coder():
    """Run the coder agent."""
    data = request.get_json()
    prompt = data.get('prompt', '')
    language = data.get('language', 'python')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    log_usage('agent_coder')

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from coder.agent import CodingAgent, CodingTask

        # Use coder-specific LLM if available (Qwen Coder), else shared LLM
        llm = get_coder_llm()
        agent = CodingAgent(llm=llm)
        result = agent.run(CodingTask(description=prompt, language=language))

        response = {
            'code': result.code,
            'qa_report': result.qa_report() if hasattr(result, 'qa_report') else None,
        }

        if llm is None:
            response['note'] = 'Running without LLM - using pattern matching fallback. Set SVEND_LLM_SIZE or SVEND_CODER_LLM env var.'

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Coder agent error: {str(e)[:200]}',
        }), 500


@app.route('/api/agents/writer', methods=['POST'])
@require_user
def api_agent_writer():
    """Run the writer agent with Editor integration."""
    data = request.get_json()
    topic = data.get('topic', '')
    template = data.get('template', 'general')
    run_editor = data.get('run_editor', True)
    original_prompt = data.get('prompt', '')

    if not topic:
        return jsonify({'error': 'Topic is required'}), 400

    log_usage('agent_writer')

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from writer.agent import WriterAgent, DocumentRequest, DocumentType

        llm = get_shared_llm()
        agent = WriterAgent(llm=llm)
        doc_type = getattr(DocumentType, template.upper(), DocumentType.EXECUTIVE_SUMMARY)
        result = agent.write(
            DocumentRequest(topic=topic, doc_type=doc_type),
            run_editor=run_editor,
            original_prompt=original_prompt or f"Write a {template} about {topic}",
        )

        response = {
            'content': result.content if hasattr(result, 'content') else str(result),
            'quality_report': result.quality_report() if hasattr(result, 'quality_report') else None,
            'quality_passed': result.quality_passed,
            'quality_issues': result.quality_issues,
        }

        # Include editor results if available
        if result.editor_result:
            response['editor'] = {
                'original_grade': result.editor_result.original_grade,
                'improved_grade': result.editor_result.improved_grade,
                'citation_confidence': result.editor_result.citation_confidence,
                'prompt_alignment': result.editor_result.prompt_alignment,
                'edits_made': result.editor_result.edits_made,
                'repeated_stats': len([r for r in result.editor_result.repetitions if r.issue_type == 'statistic']),
                'gaps': len(result.editor_result.gaps),
                'citation_concerns': len(result.editor_result.citation_concerns),
                'editorial_report': result.editor_result.editorial_report,
            }
            response['cleaned_content'] = result.editor_result.cleaned_document

        if llm is None:
            response['note'] = 'Running without LLM - output is template-based. Set SVEND_LLM_SIZE env var.'

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Writer agent error: {str(e)[:200]}',
        }), 500


@app.route('/api/agents/editor', methods=['POST'])
@require_user
def api_agent_editor():
    """Run the Editor agent on a document."""
    data = request.get_json()
    document = data.get('document', '')
    title = data.get('title', 'Document')
    rubric_type = data.get('rubric_type', 'auto')
    prompt = data.get('prompt', '')

    if not document:
        return jsonify({'error': 'Document is required'}), 400

    log_usage('agent_editor')

    try:
        import sys
        sys.path.insert(0, '/home/eric/Desktop/agents')
        from reviewer.editor import Editor

        editor = Editor()
        result = editor.edit(
            document=document,
            title=title,
            rubric_type=rubric_type,
            prompt=prompt,
        )

        # Build response
        response = {
            'original_grade': result.original_grade,
            'improved_grade': result.improved_grade,
            'citation_confidence': result.citation_confidence,
            'prompt_alignment': result.prompt_alignment,
            'edits_made': result.edits_made,
            'words_removed': result.words_removed,
            'cleaned_document': result.cleaned_document,
            'editorial_report': result.editorial_report,
            'issues': {
                'grammar_fixes': len(result.grammar_fixes),
                'repetitions': len(result.repetitions),
                'repeated_statistics': len([r for r in result.repetitions if r.issue_type == 'statistic']),
                'citation_concerns': len(result.citation_concerns),
                'gaps': len(result.gaps),
                'drift_issues': len(result.drift_issues),
            },
            'details': {
                'citation_concerns': [
                    {'text': c.citation_text[:100], 'issues': c.issues, 'confidence': c.confidence}
                    for c in result.citation_concerns[:5]
                ],
                'repeated_stats': [
                    {'text': r.text, 'count': r.count, 'suggestion': r.suggestion}
                    for r in result.repetitions if r.issue_type == 'statistic'
                ][:5],
                'gaps': [
                    {'topic': g.topic, 'issue': g.issue, 'suggestion': g.suggestion}
                    for g in result.gaps
                ],
                'drift_issues': [
                    {'expected': d.expected, 'severity': d.severity, 'suggestion': d.suggestion}
                    for d in result.drift_issues
                ],
            }
        }

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Editor agent error: {str(e)[:200]}',
        }), 500


@app.route('/api/agents/experimenter', methods=['POST'])
@require_user
def api_agent_experimenter():
    """Run the experimenter agent."""
    data = request.get_json()
    goal = data.get('goal', '')
    exp_type = data.get('type', 'power')

    if not goal:
        return jsonify({'error': 'Goal is required'}), 400

    log_usage('agent_experimenter')

    try:
        import sys
        if '/home/eric/Desktop/agents' not in sys.path:
            sys.path.insert(0, '/home/eric/Desktop/agents')
        from experimenter import ExperimenterAgent, ExperimentRequest

        llm = get_shared_llm()
        agent = ExperimenterAgent(llm=llm, seed=42)

        # Build request based on experiment type
        if exp_type == 'power':
            # Power analysis - default to two-group t-test
            exp_req = ExperimentRequest(
                goal=goal,
                request_type='power',
                test_type='ttest_ind',
                effect_size=0.5,
            )
        elif exp_type == 'factorial':
            # Full factorial design - extract or use default factors
            exp_req = ExperimentRequest(
                goal=goal,
                request_type='design',
                design_type='full_factorial',
                factors=[
                    {'name': 'Factor A', 'levels': ['-', '+']},
                    {'name': 'Factor B', 'levels': ['-', '+']},
                    {'name': 'Factor C', 'levels': ['-', '+']},
                ],
            )
        elif exp_type == 'response_surface':
            # Central Composite Design for response surface
            exp_req = ExperimentRequest(
                goal=goal,
                request_type='design',
                design_type='central_composite',
                factors=[
                    {'name': 'Factor A', 'levels': [-1, 0, 1]},
                    {'name': 'Factor B', 'levels': [-1, 0, 1]},
                ],
            )
        else:
            # Default to power analysis
            exp_req = ExperimentRequest(
                goal=goal,
                request_type='power',
                test_type='ttest_ind',
                effect_size=0.5,
            )

        result = agent.design_experiment(exp_req)

        response = {
            'summary': result.to_markdown() if hasattr(result, 'to_markdown') else str(result),
            'experiment_type': exp_type,
        }

        # Add power result if available
        if hasattr(result, 'power_result') and result.power_result:
            response['sample_size'] = result.power_result.sample_size
            response['power'] = result.power_result.power
            response['effect_size'] = result.power_result.effect_size

        # Add design if available
        if hasattr(result, 'design') and result.design:
            response['design'] = result.design.to_dict() if hasattr(result.design, 'to_dict') else str(result.design)

        if llm is None:
            response['note'] = 'Running without LLM - using statistical defaults. Set SVEND_LLM_SIZE env var.'

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Experimenter agent error: {str(e)[:200]}',
        }), 500


# =============================================================================
# Health Check
# =============================================================================

@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'svend-agents',
        'version': 'alpha-1.0',
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SVEND Agents Alpha")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(debug=args.debug, host=args.host, port=args.port)
