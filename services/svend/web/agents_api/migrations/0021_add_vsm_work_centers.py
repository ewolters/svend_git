from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agents_api', '0020_add_vsm_customers_suppliers'),
    ]

    operations = [
        migrations.AddField(
            model_name='valuestreammap',
            name='work_centers',
            field=models.JSONField(default=list, help_text='Work center groupings'),
        ),
    ]
