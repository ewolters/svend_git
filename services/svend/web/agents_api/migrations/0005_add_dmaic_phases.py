# Migration for DMAIC phase tracking

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agents_api', '0004_add_problem_model'),
    ]

    operations = [
        migrations.AddField(
            model_name='problem',
            name='methodology',
            field=models.CharField(
                choices=[
                    ('none', 'None/General'),
                    ('dmaic', 'Six Sigma DMAIC'),
                    ('doe', 'Design of Experiments'),
                    ('pdca', 'Plan-Do-Check-Act'),
                    ('a3', 'A3 Problem Solving'),
                ],
                default='none',
                help_text='Problem-solving methodology being used',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='problem',
            name='dmaic_phase',
            field=models.CharField(
                blank=True,
                choices=[
                    ('define', 'Define'),
                    ('measure', 'Measure'),
                    ('analyze', 'Analyze'),
                    ('improve', 'Improve'),
                    ('control', 'Control'),
                ],
                help_text='Current DMAIC phase (if using Six Sigma)',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='problem',
            name='phase_history',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='History of phase transitions [{phase, entered_at, notes}]',
            ),
        ),
    ]
