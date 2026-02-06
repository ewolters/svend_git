"""Populate initial schema templates."""

from django.db import migrations


def create_templates(apps, schema_editor):
    """Create default schema templates."""
    SchemaTemplate = apps.get_model("forge", "SchemaTemplate")

    templates = [
        {
            "name": "ecommerce_product",
            "domain": "ecommerce",
            "data_type": "tabular",
            "schema_def": {
                "product_id": {"type": "uuid"},
                "name": {"type": "string", "constraints": {"min_length": 5, "max_length": 100}},
                "description": {"type": "string", "constraints": {"min_length": 20, "max_length": 500}},
                "price": {"type": "float", "constraints": {"min": 0.01, "max": 10000}},
                "category": {"type": "category", "constraints": {"values": ["electronics", "clothing", "home", "sports", "books"]}},
                "in_stock": {"type": "bool"},
                "rating": {"type": "float", "constraints": {"min": 1, "max": 5}},
                "created_at": {"type": "datetime"},
            },
            "description": "E-commerce product catalog data",
        },
        {
            "name": "ecommerce_order",
            "domain": "ecommerce",
            "data_type": "tabular",
            "schema_def": {
                "order_id": {"type": "uuid"},
                "customer_id": {"type": "uuid"},
                "product_id": {"type": "uuid"},
                "quantity": {"type": "int", "constraints": {"min": 1, "max": 100}},
                "unit_price": {"type": "float", "constraints": {"min": 0.01, "max": 10000}},
                "total_price": {"type": "float", "constraints": {"min": 0.01, "max": 100000}},
                "status": {"type": "category", "constraints": {"values": ["pending", "processing", "shipped", "delivered", "cancelled"]}},
                "shipping_address": {"type": "string"},
                "created_at": {"type": "datetime"},
            },
            "description": "E-commerce order records",
        },
        {
            "name": "ecommerce_user",
            "domain": "ecommerce",
            "data_type": "tabular",
            "schema_def": {
                "user_id": {"type": "uuid"},
                "email": {"type": "email"},
                "name": {"type": "string", "constraints": {"min_length": 2, "max_length": 100}},
                "phone": {"type": "phone", "nullable": True},
                "signup_date": {"type": "date"},
                "is_premium": {"type": "bool"},
                "lifetime_value": {"type": "float", "constraints": {"min": 0, "max": 100000}},
            },
            "description": "E-commerce user profiles",
        },
        {
            "name": "customer_service_ticket",
            "domain": "customer_service",
            "data_type": "text",
            "schema_def": {
                "text_type": "ticket",
            },
            "description": "Customer support tickets with subject, body, category, and priority",
        },
        {
            "name": "customer_service_conversation",
            "domain": "customer_service",
            "data_type": "text",
            "schema_def": {
                "text_type": "conversation",
            },
            "description": "Customer support chat conversations",
        },
        {
            "name": "ecommerce_review",
            "domain": "ecommerce",
            "data_type": "text",
            "schema_def": {
                "text_type": "review",
            },
            "description": "Product reviews with ratings and sentiment",
        },
    ]

    for tmpl in templates:
        SchemaTemplate.objects.create(**tmpl)


def remove_templates(apps, schema_editor):
    """Remove default templates."""
    SchemaTemplate = apps.get_model("forge", "SchemaTemplate")
    SchemaTemplate.objects.filter(
        name__in=[
            "ecommerce_product",
            "ecommerce_order",
            "ecommerce_user",
            "customer_service_ticket",
            "customer_service_conversation",
            "ecommerce_review",
        ]
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("forge", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_templates, remove_templates),
    ]
