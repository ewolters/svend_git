from django.contrib import admin

from .models import (
    Conflict,
    GraphEdge,
    GraphEdit,
    GraphNode,
    GraphVersion,
    OperatingGraph,
    PropagationSignal,
    ProvaHypothesis,
    Trial,
    WorkingGraph,
)


class GraphNodeInline(admin.TabularInline):
    model = GraphNode
    extra = 0
    fields = ("label", "node_type", "alpha", "beta", "entity")
    readonly_fields = ("id",)


class GraphEdgeInline(admin.TabularInline):
    model = GraphEdge
    extra = 0
    fields = ("source", "target", "edge_type", "weight", "confidence", "status", "truth_frequency")
    readonly_fields = ("id",)


class GraphEditInline(admin.TabularInline):
    model = GraphEdit
    extra = 0
    fields = ("operation", "target_edge", "target_node", "params")


class TrialInline(admin.TabularInline):
    model = Trial
    extra = 0
    fields = ("status", "complexity_tier", "meets_minimum_validity", "created_at")
    readonly_fields = ("created_at",)


@admin.register(OperatingGraph)
class OperatingGraphAdmin(admin.ModelAdmin):
    list_display = ("tenant", "current_version", "predictive_score", "last_evaluated")
    list_filter = ("current_version",)
    inlines = [GraphNodeInline, GraphEdgeInline]


@admin.register(GraphVersion)
class GraphVersionAdmin(admin.ModelAdmin):
    list_display = ("operating_graph", "version_number", "promoted_by_trial", "created_at")
    list_filter = ("operating_graph",)
    readonly_fields = ("snapshot",)


@admin.register(GraphNode)
class GraphNodeAdmin(admin.ModelAdmin):
    list_display = ("label", "node_type", "operating_graph", "confidence", "alpha", "beta")
    list_filter = ("node_type", "operating_graph")
    search_fields = ("label",)


@admin.register(GraphEdge)
class GraphEdgeAdmin(admin.ModelAdmin):
    list_display = ("source", "target", "edge_type", "weight", "confidence", "status", "truth_frequency")
    list_filter = ("edge_type", "status", "operating_graph")
    search_fields = ("source__label", "target__label")


@admin.register(WorkingGraph)
class WorkingGraphAdmin(admin.ModelAdmin):
    list_display = ("owner", "tenant", "project", "operating_graph", "created_at")
    list_filter = ("tenant",)


@admin.register(ProvaHypothesis)
class ProvaHypothesisAdmin(admin.ModelAdmin):
    list_display = ("description_short", "status", "curation_tier", "prior", "posterior", "trial_commitment_date")
    list_filter = ("status",)
    inlines = [GraphEditInline, TrialInline]

    @admin.display(description="Description")
    def description_short(self, obj):
        return obj.description[:80]


@admin.register(GraphEdit)
class GraphEditAdmin(admin.ModelAdmin):
    list_display = ("hypothesis", "operation", "target_edge", "target_node")
    list_filter = ("operation",)


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ("hypothesis", "complexity_tier", "status", "meets_minimum_validity", "created_at")
    list_filter = ("status", "complexity_tier", "meets_minimum_validity")


@admin.register(Conflict)
class ConflictAdmin(admin.ModelAdmin):
    list_display = ("edge", "status", "magnitude", "evaluation_cost", "created_at")
    list_filter = ("status",)


@admin.register(PropagationSignal)
class PropagationSignalAdmin(admin.ModelAdmin):
    list_display = ("edge", "signal_type", "magnitude", "status", "created_at")
    list_filter = ("signal_type", "status")
