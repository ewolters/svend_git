from django.contrib import admin

from .models import Commitment, ModeTransition, PolicyCondition, QMSPolicy, Signal

admin.site.register(Signal)
admin.site.register(Commitment)
admin.site.register(ModeTransition)
admin.site.register(QMSPolicy)
admin.site.register(PolicyCondition)
