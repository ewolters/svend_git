# Forge Integration Plan

## Overview

Integrate Forge synthetic data generation into the SVEND Agents workbench as a new agent tab.

## Current State

### Forge Service (`/home/eric/Desktop/synthdata/`)
- Standalone FastAPI service
- Tabular + text generation
- Quality validation with Synara
- Celery workers for async processing
- Full production infrastructure (Redis, MinIO, PostgreSQL)

### ForgeClient (`/home/eric/Desktop/agents/dsw/forge_client.py`)
- Already in agents codebase
- Supports local mode (in-process) and API mode
- Used by DSW workbench

## Integration Approach

**Recommended: Direct Integration (Local Mode)**

Embed Forge generators directly into the SVEND Agents site, similar to how other agents work. This avoids running a separate service and keeps the alpha deployment simple.

## Implementation Plan

### Phase 1: Add Forge Tab to Workbench

1. **Update `home.html`**
   - Add "Forge" tab to agent selector
   - Create dynamic form for Forge options:
     - Data type: Tabular / Text
     - Domain preset: E-commerce / Customer Service / Custom
     - Record count: 100 / 500 / 1000 / 5000
     - Quality tier: Standard / Premium
     - Output format: JSON / CSV / JSONL

2. **Add `/api/agents/forge` endpoint**
   ```python
   @app.route('/api/agents/forge', methods=['POST'])
   def api_agent_forge():
       data = request.get_json()
       data_type = data.get('type', 'tabular')
       domain = data.get('domain', 'ecommerce')
       record_count = data.get('count', 100)
       quality = data.get('quality', 'standard')
       custom_schema = data.get('schema')

       # Use ForgeClient in local mode
       from dsw.forge_client import ForgeClient, ForgeConfig
       client = ForgeClient(ForgeConfig(mode='local'))

       # Get schema from domain preset or custom
       schema = get_domain_schema(domain) if not custom_schema else custom_schema

       df = client.generate(schema, n=record_count)

       return jsonify({
           'data': df.to_dict('records')[:100],  # Preview first 100
           'total_records': len(df),
           'columns': list(df.columns),
           'download_id': store_for_download(df),
       })
   ```

### Phase 2: Domain Presets

Pre-built schemas for common use cases:

```python
DOMAIN_PRESETS = {
    'ecommerce_products': {
        'product_name': {'type': 'string', 'generator': 'product_name'},
        'price': {'type': 'float', 'constraints': {'min': 0.99, 'max': 999.99}},
        'category': {'type': 'category', 'constraints': {'values': ['Electronics', 'Clothing', 'Home', 'Sports']}},
        'rating': {'type': 'float', 'constraints': {'min': 1.0, 'max': 5.0}},
        'in_stock': {'type': 'bool'},
    },
    'ecommerce_orders': {
        'order_id': {'type': 'uuid'},
        'customer_id': {'type': 'uuid'},
        'product_id': {'type': 'uuid'},
        'quantity': {'type': 'int', 'constraints': {'min': 1, 'max': 10}},
        'total_price': {'type': 'float', 'constraints': {'min': 0.99, 'max': 9999.99}},
        'status': {'type': 'category', 'constraints': {'values': ['pending', 'shipped', 'delivered', 'cancelled']}},
        'created_at': {'type': 'datetime'},
    },
    'customer_service_tickets': {
        'ticket_id': {'type': 'uuid'},
        'customer_email': {'type': 'email'},
        'subject': {'type': 'string', 'generator': 'ticket_subject'},
        'priority': {'type': 'category', 'constraints': {'values': ['low', 'medium', 'high', 'urgent']}},
        'status': {'type': 'category', 'constraints': {'values': ['open', 'in_progress', 'resolved', 'closed']}},
        'created_at': {'type': 'datetime'},
    },
    'user_profiles': {
        'user_id': {'type': 'uuid'},
        'email': {'type': 'email'},
        'name': {'type': 'string', 'generator': 'full_name'},
        'age': {'type': 'int', 'constraints': {'min': 18, 'max': 80}},
        'country': {'type': 'category', 'constraints': {'values': ['US', 'UK', 'CA', 'DE', 'FR', 'AU']}},
        'signup_date': {'type': 'datetime'},
        'is_premium': {'type': 'bool'},
    },
}
```

### Phase 3: LLM-Powered Text Generation

For text fields, use the shared Qwen LLM:

```python
def generate_text_field(field_type: str, context: dict, llm) -> str:
    """Generate realistic text using LLM."""
    prompts = {
        'product_name': f"Generate a realistic product name for category: {context.get('category', 'general')}",
        'ticket_subject': "Generate a realistic customer support ticket subject line",
        'review': f"Generate a realistic product review for: {context.get('product_name', 'product')}",
    }

    if llm and field_type in prompts:
        return llm.generate(prompts[field_type], max_tokens=50).strip()

    # Fallback to templates
    return FALLBACK_TEMPLATES.get(field_type, f"{field_type}_{uuid.uuid4().hex[:8]}")
```

### Phase 4: Download Functionality

```python
@app.route('/api/forge/download/<download_id>/<format>')
def api_forge_download(download_id, format):
    """Download generated data in requested format."""
    df = DOWNLOAD_CACHE.get(download_id)
    if not df:
        return jsonify({'error': 'Download expired'}), 404

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as f:
        if format == 'csv':
            df.to_csv(f.name, index=False)
        elif format == 'json':
            df.to_json(f.name, orient='records', indent=2)
        elif format == 'jsonl':
            df.to_json(f.name, orient='records', lines=True)

        return send_file(f.name, as_attachment=True,
                        download_name=f'synthetic_data.{format}')
```

### Phase 5: Quality Validation (Premium Tier)

For premium tier, add Synara validation:

```python
def validate_with_synara(df: pd.DataFrame, schema: dict) -> dict:
    """Run Synara quality checks on generated data."""
    from synara_test.synara import SynaraEngine

    engine = SynaraEngine()

    checks = {
        'schema_valid': check_schema_compliance(df, schema),
        'no_duplicates': df.duplicated().sum() == 0,
        'null_rate': df.isnull().sum().sum() / df.size,
        'distribution_check': check_distributions(df, schema),
    }

    # Synara reasoning for overall assessment
    assessment = engine.reason(
        f"Evaluate synthetic data quality: {json.dumps(checks)}"
    )

    return {
        'passed': all(checks.values()),
        'checks': checks,
        'synara_assessment': assessment,
    }
```

## UI Design

### Forge Tab Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Researcher] [Coder] [Writer] [Experimenter] [Forge*]      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Type:  [Tabular ▼]     Domain:  [E-commerce ▼]       │
│                                                             │
│  Preset:     [Products ▼]    Records: [1000]               │
│                                                             │
│  Quality:    ○ Standard  ● Premium (includes validation)    │
│                                                             │
│  ┌─── Custom Schema (optional) ────────────────────────┐   │
│  │ {                                                    │   │
│  │   "product_name": {"type": "string"},               │   │
│  │   "price": {"type": "float", "min": 0, "max": 100}  │   │
│  │ }                                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  [Generate Data]                                            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Output                                          [⬇ CSV]    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Generated 1000 records                               │  │
│  │                                                      │  │
│  │ Preview (first 10):                                  │  │
│  │ ┌────────────┬─────────┬──────────┬────────┐        │  │
│  │ │ product    │ price   │ category │ rating │        │  │
│  │ ├────────────┼─────────┼──────────┼────────┤        │  │
│  │ │ Widget Pro │ $49.99  │ Electronics │ 4.5 │        │  │
│  │ │ Cozy Throw │ $29.99  │ Home     │ 4.8    │        │  │
│  │ └────────────┴─────────┴──────────┴────────┘        │  │
│  │                                                      │  │
│  │ Quality Report:                                      │  │
│  │ ✓ Schema valid                                       │  │
│  │ ✓ No duplicates                                      │  │
│  │ ✓ Distributions match constraints                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

### New Files
- `site/forge_presets.py` - Domain preset schemas
- `site/templates/partials/forge_form.html` - Forge-specific form (optional)

### Modified Files
- `site/app.py` - Add `/api/agents/forge` endpoint
- `site/templates/home.html` - Add Forge tab and form

## Testing Checklist

- [ ] Generate 100 tabular records (e-commerce products)
- [ ] Generate 1000 records and download as CSV
- [ ] Custom schema generation
- [ ] LLM-powered text fields
- [ ] Premium quality validation
- [ ] Error handling for invalid schemas

## Future Enhancements

1. **Text Generation** - Full conversation/review generation with Svend
2. **Async Jobs** - Queue large requests (10K+) to background workers
3. **Templates Library** - User-saved custom schemas
4. **Data Preview** - Interactive table with sorting/filtering
5. **Schema Builder** - Visual schema designer
