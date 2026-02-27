"""Add Forge schema templates for Learn module datasets."""

from django.db import migrations


def create_templates(apps, schema_editor):
    """Create learning-focused schema templates."""
    SchemaTemplate = apps.get_model("forge", "SchemaTemplate")

    templates = [
        {
            "name": "customer_churn",
            "domain": "machine_learning",
            "data_type": "tabular",
            "description": "Telecom customer churn dataset for classification exercises. "
                           "Includes tenure, charges, service features, and churn label.",
            "schema_def": {
                "customer_id": {"type": "uuid"},
                "tenure_months": {"type": "int", "constraints": {"min": 1, "max": 72}},
                "monthly_charges": {"type": "float", "constraints": {"min": 18, "max": 120}},
                "total_charges": {"type": "float", "constraints": {"min": 18, "max": 8700}},
                "contract_type": {
                    "type": "category",
                    "constraints": {"values": ["month-to-month", "one-year", "two-year"]},
                },
                "internet_service": {
                    "type": "category",
                    "constraints": {"values": ["DSL", "fiber_optic", "none"]},
                },
                "tech_support": {"type": "bool"},
                "online_security": {"type": "bool"},
                "dependents": {"type": "bool"},
                "partner": {"type": "bool"},
                "senior_citizen": {"type": "bool"},
                "num_tickets": {"type": "int", "constraints": {"min": 0, "max": 9}},
                "avg_call_duration": {
                    "type": "float",
                    "constraints": {"min": 1, "max": 30},
                    "nullable": True,
                },
                "churned": {"type": "bool"},
            },
        },
        {
            "name": "clinical_trial",
            "domain": "advanced_statistics",
            "data_type": "tabular",
            "description": "Multi-site clinical study with repeated measures. "
                           "Includes baseline and follow-up scores, dropout, and site effects.",
            "schema_def": {
                "patient_id": {"type": "uuid"},
                "site": {
                    "type": "category",
                    "constraints": {"values": ["site_A", "site_B", "site_C", "site_D", "site_E"]},
                },
                "treatment_group": {
                    "type": "category",
                    "constraints": {"values": ["treatment", "placebo"]},
                },
                "age": {"type": "int", "constraints": {"min": 25, "max": 80}},
                "sex": {"type": "category", "constraints": {"values": ["M", "F"]}},
                "baseline_score": {"type": "float", "constraints": {"min": 20, "max": 80}},
                "week4_score": {"type": "float", "constraints": {"min": 15, "max": 85}},
                "week8_score": {
                    "type": "float",
                    "constraints": {"min": 10, "max": 90},
                    "nullable": True,
                },
                "week12_score": {
                    "type": "float",
                    "constraints": {"min": 5, "max": 95},
                    "nullable": True,
                },
                "adherence_pct": {"type": "float", "constraints": {"min": 0, "max": 100}},
                "adverse_events": {"type": "int", "constraints": {"min": 0, "max": 5}},
                "dropout": {"type": "bool"},
                "dropout_week": {
                    "type": "int",
                    "constraints": {"min": 1, "max": 12},
                    "nullable": True,
                },
            },
        },
        {
            "name": "manufacturing_quality",
            "domain": "quality_science",
            "data_type": "tabular",
            "description": "Precision manufacturing measurements for SPC and quality analysis. "
                           "Includes dimensional measurements, machine/operator info, and inspection results.",
            "schema_def": {
                "part_id": {"type": "uuid"},
                "inner_diameter": {"type": "float", "constraints": {"min": 24.95, "max": 25.05}},
                "outer_diameter": {"type": "float", "constraints": {"min": 49.95, "max": 50.05}},
                "surface_finish": {"type": "float", "constraints": {"min": 0.2, "max": 1.6}},
                "hardness": {"type": "float", "constraints": {"min": 58, "max": 65}},
                "machine_id": {
                    "type": "category",
                    "constraints": {"values": ["M1", "M2", "M3", "M4"]},
                },
                "operator": {
                    "type": "category",
                    "constraints": {"values": ["op_A", "op_B", "op_C"]},
                },
                "shift": {"type": "category", "constraints": {"values": ["day", "night"]}},
                "batch": {"type": "int", "constraints": {"min": 1, "max": 50}},
                "timestamp": {"type": "datetime"},
                "inspection_result": {
                    "type": "category",
                    "constraints": {"values": ["pass", "fail", "rework"]},
                },
            },
        },
        {
            "name": "assembly_line_production",
            "domain": "operational_excellence",
            "data_type": "tabular",
            "description": "Assembly line production data for Lean/Six Sigma exercises. "
                           "Includes cycle times, defects, downtime, and changeover metrics per station.",
            "schema_def": {
                "record_id": {"type": "uuid"},
                "date": {"type": "date"},
                "shift": {"type": "category", "constraints": {"values": ["morning", "afternoon", "night"]}},
                "station": {"type": "int", "constraints": {"min": 1, "max": 12}},
                "cycle_time_sec": {"type": "float", "constraints": {"min": 30, "max": 300}},
                "units_produced": {"type": "int", "constraints": {"min": 0, "max": 120}},
                "defects": {"type": "int", "constraints": {"min": 0, "max": 15}},
                "downtime_min": {"type": "float", "constraints": {"min": 0, "max": 60}},
                "changeover_min": {"type": "float", "constraints": {"min": 0, "max": 45}},
                "operator": {
                    "type": "category",
                    "constraints": {"values": ["team_A", "team_B", "team_C"]},
                },
                "setup_type": {
                    "type": "category",
                    "constraints": {"values": ["standard", "custom", "prototype"]},
                },
            },
        },
        {
            "name": "supply_chain_orders",
            "domain": "operations_research",
            "data_type": "tabular",
            "description": "Regional distribution network data for OR exercises. "
                           "Includes demand, shipments, costs, and stockout flags.",
            "schema_def": {
                "order_id": {"type": "uuid"},
                "date": {"type": "date"},
                "warehouse": {
                    "type": "category",
                    "constraints": {"values": ["WH_north", "WH_south", "WH_east", "WH_west",
                                               "WH_central", "WH_coastal", "WH_mountain", "WH_plains"]},
                },
                "customer_id": {"type": "uuid"},
                "product_family": {
                    "type": "category",
                    "constraints": {"values": ["family_A", "family_B", "family_C"]},
                },
                "demand_qty": {"type": "int", "constraints": {"min": 1, "max": 500}},
                "shipped_qty": {"type": "int", "constraints": {"min": 0, "max": 500}},
                "transport_cost": {"type": "float", "constraints": {"min": 5, "max": 2000}},
                "lead_time_days": {"type": "int", "constraints": {"min": 1, "max": 14}},
                "distance_km": {"type": "float", "constraints": {"min": 10, "max": 1500}},
                "stockout_flag": {"type": "bool"},
            },
        },
        {
            "name": "product_launches",
            "domain": "decision_science",
            "data_type": "tabular",
            "description": "Product launch decision scenarios for decision analysis exercises. "
                           "Includes market data, financials, risk factors, and success outcomes.",
            "schema_def": {
                "scenario_id": {"type": "uuid"},
                "market_size": {"type": "float", "constraints": {"min": 1e6, "max": 1e9}},
                "growth_rate": {"type": "float", "constraints": {"min": -0.05, "max": 0.3}},
                "competitor_count": {"type": "int", "constraints": {"min": 0, "max": 20}},
                "price_point": {"type": "float", "constraints": {"min": 5, "max": 500}},
                "development_cost": {"type": "float", "constraints": {"min": 50000, "max": 5e6}},
                "time_to_market_months": {"type": "int", "constraints": {"min": 3, "max": 36}},
                "risk_score": {"type": "float", "constraints": {"min": 0, "max": 1}},
                "npv_estimate": {"type": "float", "constraints": {"min": -1e6, "max": 1e7}},
                "success_probability": {"type": "float", "constraints": {"min": 0.05, "max": 0.95}},
            },
        },
    ]

    for tmpl in templates:
        SchemaTemplate.objects.create(**tmpl, is_builtin=True)


def remove_templates(apps, schema_editor):
    """Remove learning templates."""
    SchemaTemplate = apps.get_model("forge", "SchemaTemplate")
    SchemaTemplate.objects.filter(
        name__in=[
            "customer_churn",
            "clinical_trial",
            "manufacturing_quality",
            "assembly_line_production",
            "supply_chain_orders",
            "product_launches",
        ]
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("forge", "0004_alter_apikey_tier"),
    ]

    operations = [
        migrations.RunPython(create_templates, remove_templates),
    ]
