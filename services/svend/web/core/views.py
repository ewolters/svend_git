"""Core API views for projects, hypotheses, evidence, and knowledge graph."""

import logging
from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from accounts.permissions import rate_limited, require_ml
from .models import (
    Tenant, Membership, KnowledgeGraph, Entity, Relationship,
    Project, Dataset, ExperimentDesign, Hypothesis, Evidence, EvidenceLink,
)
from .serializers import (
    TenantSerializer, MembershipSerializer,
    KnowledgeGraphSerializer, KnowledgeGraphDetailSerializer,
    EntitySerializer, RelationshipSerializer,
    ProjectListSerializer, ProjectDetailSerializer,
    DatasetSerializer, ExperimentDesignSerializer, ExperimentDesignDetailSerializer,
    HypothesisSerializer, HypothesisDetailSerializer,
    EvidenceSerializer, EvidenceLinkSerializer,
    CreateEvidenceFromCodeSerializer, CreateEvidenceFromAnalysisSerializer,
)
from .synara import synara

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def get_user_graph(user):
    """Get or create the user's personal knowledge graph."""
    graph, created = KnowledgeGraph.objects.get_or_create(
        user=user,
        defaults={"name": f"{user.username}'s Knowledge Graph"},
    )
    return graph


def get_user_projects(user):
    """Get projects accessible to the user (personal + tenant)."""
    from django.db.models import Q

    # Personal projects
    q = Q(user=user)

    # Tenant projects (if user is member of any tenant)
    tenant_ids = Membership.objects.filter(
        user=user, is_active=True
    ).values_list("tenant_id", flat=True)

    if tenant_ids:
        q |= Q(tenant_id__in=tenant_ids)

    return Project.objects.filter(q)


# =============================================================================
# Projects
# =============================================================================

@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def project_list(request):
    """List user's projects or create a new one."""
    if request.method == "GET":
        # Note: hypothesis_count and evidence_count are model properties
        projects = get_user_projects(request.user).order_by("-updated_at")
        serializer = ProjectListSerializer(projects, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = ProjectDetailSerializer(data=request.data)
        if serializer.is_valid():
            # Assign to user's personal graph
            graph = get_user_graph(request.user)
            project = serializer.save(user=request.user, graph=graph)
            return Response(
                ProjectDetailSerializer(project).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def project_detail(request, project_id):
    """Get, update, or delete a project."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    if request.method == "GET":
        serializer = ProjectDetailSerializer(project)
        return Response(serializer.data)

    elif request.method == "PUT":
        serializer = ProjectDetailSerializer(project, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "DELETE":
        project.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def project_advance_phase(request, project_id):
    """Advance project to a new phase."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    new_phase = request.data.get("phase")
    notes = request.data.get("notes", "")

    if not new_phase:
        return Response({"error": "phase is required"}, status=status.HTTP_400_BAD_REQUEST)

    if new_phase not in dict(Project.Phase.choices):
        return Response({"error": f"Invalid phase: {new_phase}"}, status=status.HTTP_400_BAD_REQUEST)

    project.advance_phase(new_phase, notes)
    return Response(ProjectDetailSerializer(project).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def project_recalculate(request, project_id):
    """Recalculate all hypothesis probabilities in a project."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    results = synara.recalculate_project(project)
    return Response({
        "success": True,
        "results": results,
        "project": ProjectDetailSerializer(project).data,
    })


# =============================================================================
# Hypotheses
# =============================================================================

@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def hypothesis_list(request, project_id):
    """List hypotheses in a project or create a new one."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    if request.method == "GET":
        # Note: evidence_count is a model property
        hypotheses = project.hypotheses.all()
        serializer = HypothesisSerializer(hypotheses, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = HypothesisSerializer(data=request.data)
        if serializer.is_valid():
            hypothesis = serializer.save(project=project, created_by=request.user)
            return Response(
                HypothesisSerializer(hypothesis).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def hypothesis_detail(request, project_id, hypothesis_id):
    """Get, update, or delete a hypothesis."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    hypothesis = get_object_or_404(project.hypotheses, id=hypothesis_id)

    if request.method == "GET":
        serializer = HypothesisDetailSerializer(hypothesis)
        return Response(serializer.data)

    elif request.method == "PUT":
        serializer = HypothesisSerializer(hypothesis, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(HypothesisDetailSerializer(hypothesis).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "DELETE":
        hypothesis.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def hypothesis_recalculate(request, project_id, hypothesis_id):
    """Recalculate a single hypothesis probability."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    hypothesis = get_object_or_404(project.hypotheses, id=hypothesis_id)

    new_prob = synara.recalculate_hypothesis(hypothesis)
    return Response({
        "success": True,
        "probability": new_prob,
        "hypothesis": HypothesisDetailSerializer(hypothesis).data,
    })


# =============================================================================
# Evidence
# =============================================================================

@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def evidence_list(request, project_id):
    """List all evidence in a project or create new evidence."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    if request.method == "GET":
        # Get all evidence linked to any hypothesis in this project
        evidence = Evidence.objects.filter(
            hypothesis_links__hypothesis__project=project
        ).distinct()
        serializer = EvidenceSerializer(evidence, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = EvidenceSerializer(data=request.data)
        if serializer.is_valid():
            evidence = serializer.save(created_by=request.user)

            # Link to hypotheses if provided
            hypothesis_ids = request.data.get("hypothesis_ids", [])
            likelihood_ratios = request.data.get("likelihood_ratios", {})

            for hyp_id in hypothesis_ids:
                try:
                    hypothesis = project.hypotheses.get(id=hyp_id)
                    lr = likelihood_ratios.get(str(hyp_id), 1.0)
                    EvidenceLink.objects.create(
                        hypothesis=hypothesis,
                        evidence=evidence,
                        likelihood_ratio=lr,
                        is_manual=True,
                    )
                except Hypothesis.DoesNotExist:
                    pass

            return Response(
                EvidenceSerializer(evidence).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def link_evidence(request, project_id, hypothesis_id):
    """Link existing evidence to a hypothesis with a likelihood ratio."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    hypothesis = get_object_or_404(project.hypotheses, id=hypothesis_id)

    evidence_id = request.data.get("evidence_id")
    likelihood_ratio = request.data.get("likelihood_ratio", 1.0)
    reasoning = request.data.get("reasoning", "")
    apply_now = request.data.get("apply", True)

    if not evidence_id:
        return Response({"error": "evidence_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    evidence = get_object_or_404(Evidence, id=evidence_id)

    # Create or update link
    link, created = EvidenceLink.objects.update_or_create(
        hypothesis=hypothesis,
        evidence=evidence,
        defaults={
            "likelihood_ratio": likelihood_ratio,
            "reasoning": reasoning,
            "is_manual": True,
        },
    )

    # Apply Bayesian update if requested
    if apply_now:
        result = synara.apply_evidence(link)
        return Response({
            "link": EvidenceLinkSerializer(link).data,
            "update_result": {
                "prior": result.prior_probability,
                "posterior": result.posterior_probability,
                "likelihood_ratio": result.likelihood_ratio,
                "status_changed": result.status_changed,
                "new_status": result.new_status,
            },
            "hypothesis": HypothesisSerializer(hypothesis).data,
        })

    return Response({
        "link": EvidenceLinkSerializer(link).data,
        "created": created,
    }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@require_ml
def suggest_likelihood_ratio(request, project_id):
    """Get Synara's suggested likelihood ratio for evidence + hypothesis."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    evidence_id = request.data.get("evidence_id")
    hypothesis_id = request.data.get("hypothesis_id")

    if not evidence_id or not hypothesis_id:
        return Response(
            {"error": "evidence_id and hypothesis_id are required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    evidence = get_object_or_404(Evidence, id=evidence_id)
    hypothesis = get_object_or_404(project.hypotheses, id=hypothesis_id)

    suggested_lr, reasoning = synara.suggest_likelihood_ratio(evidence, hypothesis)

    return Response({
        "suggested_likelihood_ratio": suggested_lr,
        "reasoning": reasoning,
        "evidence_id": str(evidence.id),
        "hypothesis_id": str(hypothesis.id),
    })


# =============================================================================
# Evidence from Coder
# =============================================================================

@api_view(["POST"])
@permission_classes([IsAuthenticated])
@rate_limited
def create_evidence_from_code(request):
    """Create evidence from code execution results (Coder integration)."""
    serializer = CreateEvidenceFromCodeSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    project = get_object_or_404(get_user_projects(request.user), id=data["project_id"])

    with transaction.atomic():
        # Create evidence
        evidence = Evidence.objects.create(
            summary=data["summary"],
            details=data.get("details", ""),
            source_type=data.get("source_type", Evidence.SourceType.SIMULATION),
            source_description="Coder",
            result_type=Evidence.ResultType.QUANTITATIVE if data.get("p_value") else Evidence.ResultType.QUALITATIVE,
            confidence=data.get("confidence", 0.8),
            p_value=data.get("p_value"),
            effect_size=data.get("effect_size"),
            sample_size=data.get("sample_size"),
            raw_output=data.get("output", {}),
            is_reproducible=bool(data.get("code")),
            code_reference=data.get("code", ""),
            created_by=request.user,
        )

        # Link to hypotheses
        hypothesis_ids = data.get("hypothesis_ids", [])
        likelihood_ratios = data.get("likelihood_ratios", {})
        links_created = []

        for hyp_id in hypothesis_ids:
            try:
                hypothesis = project.hypotheses.get(id=hyp_id)
                lr = likelihood_ratios.get(str(hyp_id), 1.0)

                link = EvidenceLink.objects.create(
                    hypothesis=hypothesis,
                    evidence=evidence,
                    likelihood_ratio=lr,
                    is_manual=False,
                )

                # Apply Bayesian update
                result = synara.apply_evidence(link)
                links_created.append({
                    "hypothesis_id": str(hyp_id),
                    "likelihood_ratio": lr,
                    "prior": result.prior_probability,
                    "posterior": result.posterior_probability,
                })
            except Hypothesis.DoesNotExist:
                logger.warning(f"Hypothesis {hyp_id} not found in project {project.id}")

    return Response({
        "evidence": EvidenceSerializer(evidence).data,
        "links": links_created,
    }, status=status.HTTP_201_CREATED)


# =============================================================================
# Evidence from DSW Analysis
# =============================================================================

@api_view(["POST"])
@permission_classes([IsAuthenticated])
@require_ml
def create_evidence_from_analysis(request):
    """Create evidence from DSW analysis results."""
    serializer = CreateEvidenceFromAnalysisSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    project = get_object_or_404(get_user_projects(request.user), id=data["project_id"])

    results = data["results"]
    metrics = data.get("metrics", {})

    with transaction.atomic():
        # Extract statistical data from results/metrics
        p_value = metrics.get("p_value") or results.get("p_value")
        effect_size = metrics.get("effect_size") or metrics.get("r2") or metrics.get("cohen_d")
        sample_size = metrics.get("sample_size") or metrics.get("n_samples")

        # Create evidence
        evidence = Evidence.objects.create(
            summary=data["summary"],
            details=f"Analysis type: {data['analysis_type']}",
            source_type=Evidence.SourceType.ANALYSIS,
            source_description=f"DSW - {data['analysis_type']}",
            result_type=Evidence.ResultType.STATISTICAL if p_value else Evidence.ResultType.QUANTITATIVE,
            confidence=data.get("confidence", 0.8),
            p_value=p_value,
            effect_size=effect_size,
            sample_size=sample_size,
            raw_output=results,
            is_reproducible=True,
            created_by=request.user,
        )

        # Link to hypotheses
        hypothesis_ids = data.get("hypothesis_ids", [])
        likelihood_ratios = data.get("likelihood_ratios", {})
        links_created = []

        for hyp_id in hypothesis_ids:
            try:
                hypothesis = project.hypotheses.get(id=hyp_id)

                # If no LR provided, suggest one
                if str(hyp_id) not in likelihood_ratios:
                    lr, _ = synara.suggest_likelihood_ratio(evidence, hypothesis)
                else:
                    lr = likelihood_ratios[str(hyp_id)]

                link = EvidenceLink.objects.create(
                    hypothesis=hypothesis,
                    evidence=evidence,
                    likelihood_ratio=lr,
                    is_manual=False,
                )

                result = synara.apply_evidence(link)
                links_created.append({
                    "hypothesis_id": str(hyp_id),
                    "likelihood_ratio": lr,
                    "prior": result.prior_probability,
                    "posterior": result.posterior_probability,
                })
            except Hypothesis.DoesNotExist:
                pass

    return Response({
        "evidence": EvidenceSerializer(evidence).data,
        "links": links_created,
    }, status=status.HTTP_201_CREATED)


# =============================================================================
# Knowledge Graph
# =============================================================================

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def knowledge_graph(request):
    """Get the user's knowledge graph."""
    graph = get_user_graph(request.user)
    serializer = KnowledgeGraphDetailSerializer(graph)
    return Response(serializer.data)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def entity_list(request):
    """List entities or create a new one."""
    graph = get_user_graph(request.user)

    if request.method == "GET":
        entity_type = request.query_params.get("type")
        entities = graph.entities.all()
        if entity_type:
            entities = entities.filter(entity_type=entity_type)
        serializer = EntitySerializer(entities, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = EntitySerializer(data=request.data)
        if serializer.is_valid():
            entity = serializer.save(graph=graph, created_by=request.user)
            return Response(EntitySerializer(entity).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def entity_detail(request, entity_id):
    """Get, update, or delete an entity."""
    graph = get_user_graph(request.user)
    entity = get_object_or_404(graph.entities, id=entity_id)

    if request.method == "GET":
        serializer = EntitySerializer(entity)
        return Response(serializer.data)

    elif request.method == "PUT":
        serializer = EntitySerializer(entity, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "DELETE":
        entity.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def relationship_list(request):
    """List relationships or create a new one."""
    graph = get_user_graph(request.user)

    if request.method == "GET":
        relation_type = request.query_params.get("type")
        relationships = graph.relationships.all()
        if relation_type:
            relationships = relationships.filter(relation_type=relation_type)
        serializer = RelationshipSerializer(relationships, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        data = request.data.copy()
        serializer = RelationshipSerializer(data=data)
        if serializer.is_valid():
            relationship = serializer.save(graph=graph, created_by=request.user)
            return Response(RelationshipSerializer(relationship).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def check_consistency(request):
    """Check knowledge graph for logical consistency issues."""
    graph = get_user_graph(request.user)
    issues = synara.check_consistency(graph)

    return Response({
        "issues": [
            {
                "type": issue.issue_type,
                "severity": issue.severity,
                "description": issue.description,
                "entities": issue.entities_involved,
                "suggestions": issue.suggestions,
            }
            for issue in issues
        ],
        "total_issues": len(issues),
        "has_errors": any(i.severity == "error" for i in issues),
    })


# =============================================================================
# Datasets
# =============================================================================

@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def dataset_list(request, project_id):
    """List datasets in a project or upload a new one."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    if request.method == "GET":
        datasets = project.datasets.all()
        serializer = DatasetSerializer(datasets, many=True, context={"request": request})
        return Response(serializer.data)

    elif request.method == "POST":
        # Handle file upload or inline data
        name = request.data.get("name", "Untitled Dataset")
        description = request.data.get("description", "")
        data_type = request.data.get("data_type", Dataset.DataType.CSV)
        source = request.data.get("source", "")

        dataset = Dataset.objects.create(
            project=project,
            name=name,
            description=description,
            data_type=data_type,
            source=source,
            uploaded_by=request.user,
        )

        # Handle file upload
        if "file" in request.FILES:
            file = request.FILES["file"]
            dataset.file = file
            dataset.save()

            # Parse file to get columns and row count
            try:
                import pandas as pd
                import io

                if data_type == Dataset.DataType.CSV:
                    df = pd.read_csv(io.BytesIO(file.read()))
                elif data_type == Dataset.DataType.EXCEL:
                    df = pd.read_excel(io.BytesIO(file.read()))
                else:
                    df = None

                if df is not None:
                    dataset.columns = [
                        {"name": col, "type": str(df[col].dtype)}
                        for col in df.columns
                    ]
                    dataset.row_count = len(df)
                    dataset.save()
            except Exception as e:
                logger.warning(f"Failed to parse uploaded file: {e}")

        # Handle inline data
        elif "data" in request.data:
            inline_data = request.data.get("data")
            dataset.data = inline_data
            if isinstance(inline_data, list):
                dataset.row_count = len(inline_data)
                if inline_data and isinstance(inline_data[0], dict):
                    dataset.columns = [{"name": k, "type": "unknown"} for k in inline_data[0].keys()]
            dataset.save()

        return Response(
            DatasetSerializer(dataset, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def dataset_detail(request, project_id, dataset_id):
    """Get or delete a dataset."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    dataset = get_object_or_404(project.datasets, id=dataset_id)

    if request.method == "GET":
        serializer = DatasetSerializer(dataset, context={"request": request})
        data = serializer.data

        # Include data preview if inline data exists
        if dataset.data:
            data["preview"] = dataset.data[:100] if isinstance(dataset.data, list) else dataset.data

        return Response(data)

    elif request.method == "DELETE":
        if dataset.file:
            dataset.file.delete()
        dataset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dataset_data(request, project_id, dataset_id):
    """Get the full data from a dataset."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    dataset = get_object_or_404(project.datasets, id=dataset_id)

    if dataset.data:
        return Response({"data": dataset.data, "columns": dataset.columns})

    if dataset.file:
        try:
            import pandas as pd

            if dataset.data_type == Dataset.DataType.CSV:
                df = pd.read_csv(dataset.file.path)
            elif dataset.data_type == Dataset.DataType.EXCEL:
                df = pd.read_excel(dataset.file.path)
            else:
                return Response({"error": "Unsupported file type"}, status=400)

            return Response({
                "data": df.to_dict(orient="records"),
                "columns": [{"name": col, "type": str(df[col].dtype)} for col in df.columns],
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)

    return Response({"data": [], "columns": []})


# =============================================================================
# Experiment Designs
# =============================================================================

@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def experiment_design_list(request, project_id):
    """List experiment designs in a project or create a new one."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)

    if request.method == "GET":
        designs = project.experiment_designs.all()
        serializer = ExperimentDesignSerializer(designs, many=True)
        return Response(serializer.data)

    elif request.method == "POST":
        serializer = ExperimentDesignSerializer(data=request.data)
        if serializer.is_valid():
            design = serializer.save(project=project)
            return Response(
                ExperimentDesignDetailSerializer(design).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def experiment_design_detail(request, project_id, design_id):
    """Get, update, or delete an experiment design."""
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    design = get_object_or_404(project.experiment_designs, id=design_id)

    if request.method == "GET":
        serializer = ExperimentDesignDetailSerializer(design)
        return Response(serializer.data)

    elif request.method == "PUT":
        serializer = ExperimentDesignSerializer(design, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(ExperimentDesignDetailSerializer(design).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == "DELETE":
        design.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Design Execution Review
# =============================================================================

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def review_design_execution(request, project_id, design_id):
    """
    Review how well an experiment was executed against the planned design.

    POST body:
    {
        "dataset_id": "uuid" or
        "data": [...] (inline data)
    }

    Returns a comprehensive report card with:
    - Coverage Score: % of planned runs completed
    - Balance Score: Are factor levels evenly distributed?
    - Factor Fidelity: How close were actual values to planned?
    - Randomization Quality: Was run order respected?
    - Sample Quality: Outliers, missing values, variance
    - Overall Execution Score: Weighted combination
    """
    project = get_object_or_404(get_user_projects(request.user), id=project_id)
    design = get_object_or_404(project.experiment_designs, id=design_id)

    # Get actual data
    actual_data = None
    dataset = None

    if "dataset_id" in request.data:
        dataset = get_object_or_404(project.datasets, id=request.data["dataset_id"])
        if dataset.data:
            actual_data = dataset.data
        elif dataset.file:
            try:
                import pandas as pd
                if dataset.data_type == Dataset.DataType.CSV:
                    df = pd.read_csv(dataset.file.path)
                else:
                    df = pd.read_excel(dataset.file.path)
                actual_data = df.to_dict(orient="records")
            except Exception as e:
                return Response({"error": f"Failed to read dataset: {e}"}, status=400)

    elif "data" in request.data:
        actual_data = request.data["data"]

    if not actual_data:
        return Response({"error": "No data provided"}, status=400)

    # Run the execution review
    try:
        review_result = _perform_execution_review(design, actual_data)
    except Exception as e:
        logger.exception("Execution review failed")
        return Response({"error": str(e)}, status=500)

    # Save review to design
    design.execution_review = review_result
    design.execution_score = review_result["overall_score"]
    design.status = ExperimentDesign.Status.REVIEWED
    design.save()

    # If linked to dataset, update the reference
    if dataset:
        dataset.experiment_design = design
        dataset.save()

    return Response({
        "success": True,
        "review": review_result,
        "design": ExperimentDesignDetailSerializer(design).data,
    })


def _perform_execution_review(design, actual_data):
    """
    Perform comprehensive execution review.

    Metrics:
    - Coverage (30%): Did all planned runs get completed?
    - Balance (20%): Are factor levels evenly distributed in actual data?
    - Fidelity (25%): How close are actual values to planned values?
    - Randomization (10%): Was the randomized order followed?
    - Sample Quality (15%): Data quality (outliers, variance, completeness)
    """
    import numpy as np
    from collections import Counter

    design_spec = design.design_spec
    planned_runs = design_spec.get("runs", [])
    factors = design.factors

    n_planned = len(planned_runs)
    n_actual = len(actual_data)

    issues = []
    recommendations = []

    # 1. COVERAGE SCORE (30%)
    # Check if all planned runs are present in actual data
    coverage_score = min(100, (n_actual / n_planned * 100)) if n_planned > 0 else 0

    if n_actual < n_planned:
        missing = n_planned - n_actual
        issues.append(f"Missing {missing} of {n_planned} planned runs ({missing/n_planned*100:.1f}%)")
        recommendations.append("Ensure all planned experimental runs are completed before analysis")
    elif n_actual > n_planned:
        extra = n_actual - n_planned
        issues.append(f"Found {extra} extra runs beyond the {n_planned} planned")
        coverage_score = 100  # Don't penalize for extra runs

    # 2. BALANCE SCORE (20%)
    # Check if factor levels are evenly distributed
    balance_scores = []

    for factor in factors:
        factor_name = factor.get("name")
        planned_levels = factor.get("levels", [])

        # Count actual level distribution
        actual_levels = [row.get(factor_name) for row in actual_data if factor_name in row]
        if not actual_levels:
            continue

        level_counts = Counter(actual_levels)

        # Expected count per level
        expected_per_level = len(actual_levels) / len(planned_levels) if planned_levels else len(actual_levels)

        # Calculate balance (chi-square-like metric)
        if expected_per_level > 0:
            deviations = []
            for level in planned_levels:
                actual_count = level_counts.get(level, 0)
                deviation = abs(actual_count - expected_per_level) / expected_per_level
                deviations.append(deviation)

            avg_deviation = np.mean(deviations) if deviations else 0
            factor_balance = max(0, 100 - avg_deviation * 100)
            balance_scores.append(factor_balance)

            if avg_deviation > 0.2:
                issues.append(f"Imbalanced levels for {factor_name}: {dict(level_counts)}")

    balance_score = np.mean(balance_scores) if balance_scores else 100

    # 3. FIDELITY SCORE (25%)
    # How close are actual factor values to planned values?
    fidelity_scores = []

    # Try to match actual runs to planned runs
    for factor in factors:
        factor_name = factor.get("name")
        planned_levels = set(factor.get("levels", []))

        actual_values = [row.get(factor_name) for row in actual_data if factor_name in row]

        # Check if actual values match planned levels
        out_of_spec = 0
        for val in actual_values:
            if val not in planned_levels:
                # For numeric factors, check if within tolerance
                try:
                    val_float = float(val)
                    min_level = min(float(l) for l in planned_levels)
                    max_level = max(float(l) for l in planned_levels)
                    range_size = max_level - min_level if max_level != min_level else 1

                    # 10% tolerance outside range
                    if val_float < min_level - 0.1 * range_size or val_float > max_level + 0.1 * range_size:
                        out_of_spec += 1
                except (ValueError, TypeError):
                    # Categorical - exact match required
                    out_of_spec += 1

        if actual_values:
            fidelity = (1 - out_of_spec / len(actual_values)) * 100
            fidelity_scores.append(fidelity)

            if out_of_spec > 0:
                issues.append(f"{out_of_spec} runs have {factor_name} outside planned levels")
                recommendations.append(f"Review {factor_name} settings - ensure factor controller is calibrated")

    fidelity_score = np.mean(fidelity_scores) if fidelity_scores else 100

    # 4. RANDOMIZATION SCORE (10%)
    # Check if runs were executed in randomized order
    randomization_score = 100  # Start optimistic

    # Try to match run order
    if "run_order" in actual_data[0] if actual_data else False:
        actual_orders = [row.get("run_order") for row in actual_data]
        planned_orders = [run.get("run_order") for run in planned_runs]

        if actual_orders == sorted(actual_orders):
            # Runs were done in sequential order, not randomized
            randomization_score = 50
            issues.append("Runs appear to be executed in standard order, not randomized")
            recommendations.append("Execute runs in the randomized order to avoid systematic bias")

    # 5. SAMPLE QUALITY SCORE (15%)
    # Check for outliers, variance, completeness
    quality_scores = []

    # Get response column(s)
    response_names = [r.get("name", "Response") for r in design.responses] if design.responses else ["Response"]

    for resp_name in response_names:
        response_values = []
        for row in actual_data:
            val = row.get(resp_name) or row.get("response") or row.get("Response")
            if val is not None:
                try:
                    response_values.append(float(val))
                except (ValueError, TypeError):
                    pass

        if len(response_values) < 2:
            quality_scores.append(50)  # Can't assess quality with too few values
            issues.append(f"Insufficient response data for {resp_name}")
            continue

        # Check for missing values
        missing_pct = (n_actual - len(response_values)) / n_actual * 100 if n_actual > 0 else 0
        if missing_pct > 0:
            issues.append(f"{missing_pct:.1f}% missing values in {resp_name}")

        # Check for outliers (IQR method)
        q1, q3 = np.percentile(response_values, [25, 75])
        iqr = q3 - q1
        outliers = [v for v in response_values if v < q1 - 1.5*iqr or v > q3 + 1.5*iqr]
        outlier_pct = len(outliers) / len(response_values) * 100

        if outlier_pct > 10:
            issues.append(f"{len(outliers)} potential outliers detected in {resp_name} ({outlier_pct:.1f}%)")
            recommendations.append("Review outlier runs for measurement errors or unusual conditions")

        # Check variance (coefficient of variation)
        cv = np.std(response_values) / np.mean(response_values) if np.mean(response_values) != 0 else 0

        # Quality score based on completeness and outlier presence
        completeness = (1 - missing_pct / 100) * 50
        outlier_penalty = min(50, outlier_pct * 2)
        quality = completeness + (50 - outlier_penalty)
        quality_scores.append(quality)

    sample_quality_score = np.mean(quality_scores) if quality_scores else 100

    # CALCULATE OVERALL SCORE (weighted)
    weights = {
        "coverage": 0.30,
        "balance": 0.20,
        "fidelity": 0.25,
        "randomization": 0.10,
        "sample_quality": 0.15,
    }

    overall_score = (
        coverage_score * weights["coverage"] +
        balance_score * weights["balance"] +
        fidelity_score * weights["fidelity"] +
        randomization_score * weights["randomization"] +
        sample_quality_score * weights["sample_quality"]
    )

    # Generate grade
    if overall_score >= 90:
        grade = "A"
        grade_description = "Excellent execution - results are highly reliable"
    elif overall_score >= 80:
        grade = "B"
        grade_description = "Good execution - results are reliable with minor concerns"
    elif overall_score >= 70:
        grade = "C"
        grade_description = "Acceptable execution - results usable but consider noted issues"
    elif overall_score >= 60:
        grade = "D"
        grade_description = "Poor execution - significant issues may affect validity"
    else:
        grade = "F"
        grade_description = "Failed execution - results may not be reliable"

    return {
        "overall_score": round(overall_score, 1),
        "grade": grade,
        "grade_description": grade_description,
        "scores": {
            "coverage": {
                "score": round(coverage_score, 1),
                "weight": weights["coverage"],
                "description": "Completion of planned experimental runs",
            },
            "balance": {
                "score": round(balance_score, 1),
                "weight": weights["balance"],
                "description": "Even distribution of factor levels",
            },
            "fidelity": {
                "score": round(fidelity_score, 1),
                "weight": weights["fidelity"],
                "description": "Adherence to planned factor values",
            },
            "randomization": {
                "score": round(randomization_score, 1),
                "weight": weights["randomization"],
                "description": "Following randomized run order",
            },
            "sample_quality": {
                "score": round(sample_quality_score, 1),
                "weight": weights["sample_quality"],
                "description": "Data completeness and outlier assessment",
            },
        },
        "summary": {
            "planned_runs": n_planned,
            "actual_runs": n_actual,
            "factors_checked": len(factors),
        },
        "issues": issues,
        "recommendations": recommendations,
    }
