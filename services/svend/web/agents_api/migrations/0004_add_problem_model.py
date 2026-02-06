# Generated manually for Problem model - Decision Science Workbench

import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agents_api', '0003_add_saved_models'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Problem',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=255)),
                ('status', models.CharField(choices=[('active', 'Active'), ('resolved', 'Resolved'), ('abandoned', 'Abandoned')], default='active', max_length=20)),
                # Effect
                ('effect_description', models.TextField(help_text='What are you observing? The symptom, outcome, or situation.')),
                ('effect_magnitude', models.CharField(blank=True, help_text="How big is the effect? (e.g., '40% increase', 'severe', '$50k loss')", max_length=100)),
                ('effect_first_observed', models.CharField(blank=True, help_text='When did you first notice this?', max_length=100)),
                ('effect_confidence', models.CharField(default='medium', help_text='How confident are you that this effect is real?', max_length=20)),
                # Context
                ('domain', models.CharField(blank=True, help_text="Domain area (e.g., 'manufacturing', 'SaaS', 'healthcare')", max_length=100)),
                ('stakeholders', models.JSONField(blank=True, default=list, help_text='Who is affected or involved?')),
                ('constraints', models.JSONField(blank=True, default=list, help_text='Constraints on the investigation or solution')),
                ('prior_beliefs', models.JSONField(blank=True, default=list, help_text='Initial beliefs about the cause [{belief, confidence}]')),
                ('can_experiment', models.BooleanField(default=True, help_text='Can you run controlled experiments?')),
                ('available_data', models.TextField(blank=True, help_text='What data do you have access to?')),
                # Living state
                ('hypotheses', models.JSONField(blank=True, default=list, help_text='List of causal hypotheses being investigated')),
                ('evidence', models.JSONField(blank=True, default=list, help_text='Evidence gathered from research, analysis, experiments')),
                ('dead_ends', models.JSONField(blank=True, default=list, help_text="Hypotheses we've ruled out")),
                # Understanding
                ('probable_causes', models.JSONField(blank=True, default=list, help_text='Most likely causes [{cause, probability, confidence}]')),
                ('key_uncertainties', models.JSONField(blank=True, default=list, help_text="What we still don't know")),
                ('recommended_next_steps', models.JSONField(blank=True, default=list, help_text='Suggested next actions')),
                # Bias tracking
                ('bias_warnings', models.JSONField(blank=True, default=list, help_text='Cognitive biases detected [{type, description, timestamp}]')),
                # Resolution
                ('resolution_summary', models.TextField(blank=True, help_text='What did we learn? What was the probable cause?')),
                ('resolution_confidence', models.CharField(blank=True, help_text='Confidence in the resolution', max_length=20)),
                # Timestamps
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                # Foreign key
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='problems', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-updated_at'],
            },
        ),
        migrations.AddIndex(
            model_name='problem',
            index=models.Index(fields=['user', 'status'], name='agents_api__user_id_c8e7a5_idx'),
        ),
        migrations.AddIndex(
            model_name='problem',
            index=models.Index(fields=['user', '-updated_at'], name='agents_api__user_id_f3a2d1_idx'),
        ),
    ]
