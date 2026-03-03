"""
Research Validator

Quality checks for research output:
- Hallucination detection (claims must trace to sources)
- Source consistency checking (flag contradictions)
- Confidence justification (explain why high/low)
- Quality gates (block bad research)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher


@dataclass
class ClaimValidation:
    """Validation result for a single claim."""
    claim: str
    is_supported: bool
    supporting_sources: list[str]  # Source IDs that support this claim
    contradicting_sources: list[str]  # Source IDs that contradict
    confidence: float  # 0-1 how confident the claim is
    issues: list[str]  # Problems found

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0 or not self.is_supported


@dataclass
class SourceConsistency:
    """Result of checking consistency across sources."""
    is_consistent: bool
    contradictions: list[dict]  # {claim, source1, source2, details}
    agreements: list[dict]  # {claim, sources, confidence}
    consistency_score: float  # 0-1


@dataclass
class ConfidenceJustification:
    """Explanation of why confidence is at a certain level."""
    level: str  # "high", "medium", "low"
    score: float  # 0-1
    positive_factors: list[str]
    negative_factors: list[str]
    recommendation: str  # What would improve confidence


@dataclass
class ValidationResult:
    """Complete validation result for research output."""
    is_valid: bool
    overall_quality: float  # 0-1

    # Claim validation
    claims_checked: int
    claims_supported: int
    claims_unsupported: int
    claim_validations: list[ClaimValidation]

    # Source consistency
    source_consistency: SourceConsistency

    # Confidence justification
    confidence_justification: ConfidenceJustification

    # Issues
    critical_issues: list[str]  # Must fix
    warnings: list[str]  # Should fix

    # Quality gates
    passed_gates: list[str]
    failed_gates: list[str]

    def summary(self) -> str:
        """Generate validation summary."""
        lines = [
            "# Research Validation Report",
            "",
            f"**Overall Quality:** {self.overall_quality:.0%}",
            f"**Valid:** {'YES' if self.is_valid else 'NO'}",
            "",
            "## Claim Validation",
            f"- Claims checked: {self.claims_checked}",
            f"- Supported: {self.claims_supported}",
            f"- Unsupported: {self.claims_unsupported}",
            "",
        ]

        if self.claim_validations:
            unsupported = [c for c in self.claim_validations if not c.is_supported]
            if unsupported:
                lines.append("### Unsupported Claims")
                for claim in unsupported[:5]:
                    lines.append(f"- \"{claim.claim[:60]}...\"")
                    for issue in claim.issues:
                        lines.append(f"  - {issue}")
                lines.append("")

        lines.extend([
            "## Source Consistency",
            f"- Consistency Score: {self.source_consistency.consistency_score:.0%}",
            f"- Contradictions Found: {len(self.source_consistency.contradictions)}",
            "",
        ])

        if self.source_consistency.contradictions:
            lines.append("### Contradictions")
            for c in self.source_consistency.contradictions[:3]:
                lines.append(f"- {c['details']}")
            lines.append("")

        lines.extend([
            "## Confidence Justification",
            f"**Level:** {self.confidence_justification.level.upper()} ({self.confidence_justification.score:.0%})",
            "",
            "**Positive Factors:**",
        ])
        for f in self.confidence_justification.positive_factors:
            lines.append(f"- {f}")

        lines.append("")
        lines.append("**Negative Factors:**")
        for f in self.confidence_justification.negative_factors:
            lines.append(f"- {f}")

        lines.extend([
            "",
            f"**Recommendation:** {self.confidence_justification.recommendation}",
            "",
        ])

        if self.critical_issues:
            lines.append("## Critical Issues")
            for issue in self.critical_issues:
                lines.append(f"- {issue}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("## Quality Gates")
        lines.append(f"- Passed: {len(self.passed_gates)}")
        lines.append(f"- Failed: {len(self.failed_gates)}")

        if self.failed_gates:
            lines.append("")
            lines.append("### Failed Gates")
            for gate in self.failed_gates:
                lines.append(f"- {gate}")

        return "\n".join(lines)


class ResearchValidator:
    """
    Validates research output for quality and accuracy.

    Checks:
    1. Hallucination - Claims must trace to source text
    2. Consistency - Sources shouldn't contradict each other
    3. Confidence - Explains why confidence is high/low
    4. Quality Gates - Must pass minimum thresholds
    """

    # Minimum thresholds for quality gates
    MIN_SOURCES = 3
    MIN_SOURCE_TYPES = 2
    MIN_CLAIM_SUPPORT = 0.5  # At least 50% of claims must be supported
    MIN_CONSISTENCY = 0.6
    MIN_CONFIDENCE = 0.4  # Below this, fail the research

    # Words that indicate uncertainty
    UNCERTAINTY_MARKERS = [
        'might', 'may', 'could', 'possibly', 'perhaps', 'likely',
        'unlikely', 'suggests', 'appears', 'seems', 'unclear',
        'uncertain', 'debated', 'controversial', 'some argue',
        'it is thought', 'believed to', 'estimated'
    ]

    # Words that indicate strong claims
    STRONG_CLAIM_MARKERS = [
        'proves', 'definitely', 'certainly', 'always', 'never',
        'all', 'none', 'guaranteed', 'absolutely', 'undoubtedly',
        'without question', 'conclusively', 'unquestionably'
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, apply stricter thresholds
        """
        self.strict_mode = strict_mode
        if strict_mode:
            self.MIN_CLAIM_SUPPORT = 0.7
            self.MIN_CONSISTENCY = 0.8
            self.MIN_CONFIDENCE = 0.6

    def validate(self, synthesis: dict, findings: list[dict],
                 sources: list) -> ValidationResult:
        """
        Validate research output.

        Args:
            synthesis: The synthesized output from _synthesize()
            findings: List of {query, source, finding} dicts
            sources: List of Source objects

        Returns:
            ValidationResult with all checks
        """
        # Extract claims from synthesis
        claims = self._extract_claims(synthesis)

        # Build source text index for claim checking
        source_texts = {}
        for f in findings:
            src_id = f["source"].id
            if src_id not in source_texts:
                source_texts[src_id] = []
            source_texts[src_id].append(f["finding"])

        # 1. Validate claims (hallucination detection)
        claim_validations = []
        for claim in claims:
            validation = self._validate_claim(claim, source_texts, sources)
            claim_validations.append(validation)

        claims_supported = sum(1 for c in claim_validations if c.is_supported)
        claims_unsupported = len(claim_validations) - claims_supported

        # 2. Check source consistency
        source_consistency = self._check_consistency(findings, sources)

        # 3. Generate confidence justification
        confidence_justification = self._justify_confidence(
            synthesis, findings, sources, claim_validations, source_consistency
        )

        # 4. Run quality gates
        passed_gates, failed_gates = self._run_quality_gates(
            sources, claim_validations, source_consistency, confidence_justification
        )

        # 5. Collect issues
        critical_issues = []
        warnings = []

        # Check for unsupported strong claims
        for cv in claim_validations:
            if not cv.is_supported:
                if any(m in cv.claim.lower() for m in self.STRONG_CLAIM_MARKERS):
                    critical_issues.append(f"Unsupported strong claim: \"{cv.claim[:50]}...\"")
                else:
                    warnings.append(f"Claim not directly supported: \"{cv.claim[:50]}...\"")

        # Check for contradictions
        if source_consistency.contradictions:
            for c in source_consistency.contradictions[:3]:
                warnings.append(f"Source contradiction: {c['details'][:80]}")

        # Low source diversity
        source_types = set(s.source_type for s in sources)
        if len(source_types) < self.MIN_SOURCE_TYPES:
            warnings.append(f"Low source diversity: only {len(source_types)} source type(s)")

        # Calculate overall quality
        claim_quality = claims_supported / len(claim_validations) if claim_validations else 0.5
        overall_quality = (
            claim_quality * 0.4 +
            source_consistency.consistency_score * 0.3 +
            confidence_justification.score * 0.3
        )

        # Determine validity
        is_valid = (
            len(failed_gates) == 0 and
            len(critical_issues) == 0 and
            overall_quality >= 0.5
        )

        return ValidationResult(
            is_valid=is_valid,
            overall_quality=overall_quality,
            claims_checked=len(claim_validations),
            claims_supported=claims_supported,
            claims_unsupported=claims_unsupported,
            claim_validations=claim_validations,
            source_consistency=source_consistency,
            confidence_justification=confidence_justification,
            critical_issues=critical_issues,
            warnings=warnings,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
        )

    def _extract_claims(self, synthesis: dict) -> list[str]:
        """Extract claims from synthesis for validation."""
        claims = []

        # From summary
        summary = synthesis.get("summary", "")
        if summary:
            # Split into sentences
            sentences = re.split(r'[.!?]+', summary)
            claims.extend(s.strip() for s in sentences if len(s.strip()) > 20)

        # From key_findings
        for finding in synthesis.get("key_findings", []):
            claims.append(finding)

        # From evidence
        for evidence in synthesis.get("evidence", []):
            claim = evidence.get("claim", "")
            if claim:
                claims.append(claim)

        return claims

    def _validate_claim(self, claim: str, source_texts: dict,
                        sources: list) -> ClaimValidation:
        """
        Validate a single claim against source texts.

        A claim is supported if:
        1. Key terms from the claim appear in at least one source
        2. The semantic meaning is similar (windowed fuzzy matching)
        3. Bigram overlap confirms phrase-level matches
        """
        claim_lower = claim.lower()

        # Extract key terms (nouns, numbers, significant words)
        key_terms = self._extract_key_terms(claim)

        # Extract bigrams for phrase-level matching
        claim_bigrams = self._extract_bigrams(claim_lower)

        supporting = []
        contradicting = []
        best_scores = []  # Track best match score per source
        issues = []

        for src_id, texts in source_texts.items():
            combined_text = " ".join(texts).lower()

            # Check term overlap
            terms_found = sum(1 for t in key_terms if t in combined_text)
            term_coverage = terms_found / len(key_terms) if key_terms else 0

            # Check bigram overlap (phrase-level matching)
            bigram_score = 0.0
            if claim_bigrams:
                src_bigrams = self._extract_bigrams(combined_text)
                if src_bigrams:
                    common = claim_bigrams & src_bigrams
                    bigram_score = len(common) / len(claim_bigrams)

            # Check windowed fuzzy similarity (compare claim against
            # same-length windows of source, not the entire source)
            max_similarity = 0
            for text in texts:
                similarity = self._fuzzy_similarity(claim_lower, text.lower())
                max_similarity = max(max_similarity, similarity)

            # Combined score: weighted average of all signals
            combined = (
                0.4 * term_coverage
                + 0.3 * bigram_score
                + 0.3 * max_similarity
            )
            best_scores.append(combined)

            # Support thresholds: any strong signal or combined above 0.35
            if term_coverage >= 0.5 or max_similarity >= 0.4 or combined >= 0.35:
                supporting.append(src_id)

            # Check for contradictions (negation patterns)
            if self._check_contradiction(claim_lower, combined_text):
                contradicting.append(src_id)

        # Determine if supported
        is_supported = len(supporting) >= 1

        if not is_supported:
            issues.append("Claim not found in any source text")

        if contradicting:
            issues.append(f"Contradicted by {len(contradicting)} source(s)")

        # Check for overconfident language without support
        if any(m in claim_lower for m in self.STRONG_CLAIM_MARKERS):
            if len(supporting) < 2:
                issues.append("Strong claim with insufficient source support")

        # Confidence: smooth curve that rewards multiple sources
        # 0 sources → 0.0, 1 → 0.5, 2 → 0.7, 3 → 0.82, 5+ → ~0.95
        support_count = len(supporting)
        if support_count == 0:
            confidence = 0.0
        else:
            confidence = 1.0 - 0.5 ** support_count
            # Boost by best match quality
            if best_scores:
                best_match = max(best_scores)
                confidence = confidence * 0.7 + best_match * 0.3
        if contradicting:
            confidence *= 0.5

        return ClaimValidation(
            claim=claim,
            is_supported=is_supported,
            supporting_sources=supporting,
            contradicting_sources=contradicting,
            confidence=confidence,
            issues=issues,
        )

    def _extract_bigrams(self, text: str) -> set:
        """Extract word bigrams for phrase-level matching."""
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms for matching."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
            'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
            'or', 'because', 'until', 'while', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our',
            'you', 'your', 'he', 'she', 'his', 'her', 'which', 'who'
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter and return
        key_terms = [w for w in words if w not in stop_words]

        # Also extract numbers and percentages
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        key_terms.extend(numbers)

        return key_terms[:20]  # Limit

    def _fuzzy_similarity(self, claim: str, source: str) -> float:
        """Calculate fuzzy similarity using windowed comparison.

        Instead of comparing a short claim against an entire source text
        (which always scores low), slide a claim-sized window across the
        source and return the best match.
        """
        claim_len = len(claim)
        source_len = len(source)

        if source_len <= claim_len * 2:
            # Source is short enough to compare directly
            return SequenceMatcher(None, claim, source).ratio()

        # Slide a window of claim_len * 1.5 across the source
        window = int(claim_len * 1.5)
        step = max(1, window // 4)
        best = 0.0

        for i in range(0, source_len - window + 1, step):
            chunk = source[i:i + window]
            score = SequenceMatcher(None, claim, chunk).ratio()
            if score > best:
                best = score
                if best > 0.7:
                    break  # Good enough, stop early

        return best

    def _check_contradiction(self, claim: str, source_text: str) -> bool:
        """Check if source text contradicts the claim."""
        # Simple heuristic: look for negation patterns
        negation_patterns = [
            r'not\s+' + re.escape(claim[:20]),
            r'no\s+evidence',
            r'contrary\s+to',
            r'however.*(?:not|no|never)',
            r'(?:false|incorrect|wrong|inaccurate)',
        ]

        for pattern in negation_patterns:
            if re.search(pattern, source_text, re.IGNORECASE):
                return True

        return False

    def _check_consistency(self, findings: list[dict],
                           sources: list) -> SourceConsistency:
        """Check consistency across sources."""
        contradictions = []
        agreements = []

        # Group findings by topic/query
        by_query = {}
        for f in findings:
            query = f["query"]
            if query not in by_query:
                by_query[query] = []
            by_query[query].append(f)

        # Check each group for contradictions
        for query, query_findings in by_query.items():
            if len(query_findings) < 2:
                continue

            # Compare findings pairwise
            for i, f1 in enumerate(query_findings):
                for f2 in query_findings[i+1:]:
                    text1 = f1["finding"].lower()
                    text2 = f2["finding"].lower()

                    # Check for contradicting statements
                    if self._texts_contradict(text1, text2):
                        contradictions.append({
                            "claim": query,
                            "source1": f1["source"].id,
                            "source2": f2["source"].id,
                            "details": f"'{f1['source'].title}' vs '{f2['source'].title}' on: {query[:50]}",
                        })

                    # Check for agreement
                    similarity = self._fuzzy_similarity(text1, text2)
                    if similarity > 0.5:
                        agreements.append({
                            "claim": query,
                            "sources": [f1["source"].id, f2["source"].id],
                            "confidence": similarity,
                        })

        # Calculate consistency score
        total_comparisons = sum(len(f) * (len(f) - 1) // 2 for f in by_query.values())
        if total_comparisons == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (len(contradictions) / total_comparisons)

        return SourceConsistency(
            is_consistent=len(contradictions) == 0,
            contradictions=contradictions,
            agreements=agreements,
            consistency_score=max(0, consistency_score),
        )

    def _texts_contradict(self, text1: str, text2: str) -> bool:
        """Check if two texts contradict each other."""
        # Look for opposing numbers
        nums1 = set(re.findall(r'\d+(?:\.\d+)?%?', text1))
        nums2 = set(re.findall(r'\d+(?:\.\d+)?%?', text2))

        # If both have numbers but they're very different
        if nums1 and nums2:
            for n1 in nums1:
                for n2 in nums2:
                    try:
                        v1 = float(n1.rstrip('%'))
                        v2 = float(n2.rstrip('%'))
                        # If values differ by more than 50%
                        if abs(v1 - v2) / max(v1, v2, 1) > 0.5:
                            return True
                    except ValueError:
                        pass

        # Look for explicit contradiction markers
        contradiction_patterns = [
            (r'increase', r'decrease'),
            (r'growth', r'decline'),
            (r'positive', r'negative'),
            (r'success', r'failure'),
            (r'support', r'oppose'),
            (r'safe', r'dangerous'),
            (r'effective', r'ineffective'),
        ]

        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, text1) and re.search(pattern2, text2):
                return True
            if re.search(pattern2, text1) and re.search(pattern1, text2):
                return True

        return False

    def _justify_confidence(self, synthesis: dict, findings: list[dict],
                            sources: list, claim_validations: list[ClaimValidation],
                            consistency: SourceConsistency) -> ConfidenceJustification:
        """Generate explanation for confidence level."""
        positive_factors = []
        negative_factors = []

        # Source count
        if len(sources) >= 5:
            positive_factors.append(f"Good source coverage ({len(sources)} sources)")
        elif len(sources) >= 3:
            positive_factors.append(f"Adequate source coverage ({len(sources)} sources)")
        else:
            negative_factors.append(f"Limited sources ({len(sources)})")

        # Source diversity
        source_types = set(s.source_type.value if hasattr(s.source_type, 'value') else str(s.source_type) for s in sources)
        if len(source_types) >= 3:
            positive_factors.append(f"High source diversity ({len(source_types)} types)")
        elif len(source_types) >= 2:
            positive_factors.append(f"Moderate source diversity ({len(source_types)} types)")
        else:
            negative_factors.append("Limited source diversity")

        # Claim support
        if claim_validations:
            support_rate = sum(1 for c in claim_validations if c.is_supported) / len(claim_validations)
            if support_rate >= 0.8:
                positive_factors.append(f"High claim support rate ({support_rate:.0%})")
            elif support_rate >= 0.6:
                positive_factors.append(f"Moderate claim support ({support_rate:.0%})")
            else:
                negative_factors.append(f"Low claim support rate ({support_rate:.0%})")

        # Consistency
        if consistency.consistency_score >= 0.9:
            positive_factors.append("Sources are highly consistent")
        elif consistency.consistency_score >= 0.7:
            positive_factors.append("Sources are mostly consistent")
        else:
            negative_factors.append(f"Source inconsistency detected ({consistency.consistency_score:.0%})")

        if consistency.contradictions:
            negative_factors.append(f"{len(consistency.contradictions)} source contradiction(s)")

        # Academic sources
        academic_count = sum(1 for s in sources if 'academic' in str(s.source_type).lower())
        if academic_count >= 2:
            positive_factors.append(f"{academic_count} academic/peer-reviewed sources")
        elif academic_count == 0:
            negative_factors.append("No academic sources")

        # Gaps identified
        gaps = synthesis.get("gaps", [])
        if gaps:
            negative_factors.append(f"Research gaps identified ({len(gaps)})")

        # Conflicts identified
        conflicts = synthesis.get("conflicts", [])
        if conflicts:
            negative_factors.append(f"Conflicting information ({len(conflicts)} areas)")

        # Calculate score
        score = 0.5  # Start at medium
        score += len(positive_factors) * 0.1
        score -= len(negative_factors) * 0.1
        score = max(0, min(1, score))

        # Determine level
        if score >= 0.75:
            level = "high"
        elif score >= 0.5:
            level = "medium"
        else:
            level = "low"

        # Recommendation
        if score < 0.5:
            if len(sources) < 3:
                recommendation = "Add more sources to improve confidence"
            elif consistency.contradictions:
                recommendation = "Resolve source contradictions before finalizing"
            else:
                recommendation = "Verify claims with additional primary sources"
        elif score < 0.75:
            recommendation = "Consider adding academic sources for higher confidence"
        else:
            recommendation = "Research quality is good - proceed with synthesis"

        return ConfidenceJustification(
            level=level,
            score=score,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            recommendation=recommendation,
        )

    def _run_quality_gates(self, sources: list,
                           claim_validations: list[ClaimValidation],
                           consistency: SourceConsistency,
                           confidence: ConfidenceJustification) -> tuple[list[str], list[str]]:
        """Run quality gates and return (passed, failed) lists."""
        passed = []
        failed = []

        # Gate 1: Minimum sources
        if len(sources) >= self.MIN_SOURCES:
            passed.append(f"Minimum sources ({len(sources)} >= {self.MIN_SOURCES})")
        else:
            failed.append(f"Insufficient sources ({len(sources)} < {self.MIN_SOURCES})")

        # Gate 2: Source type diversity
        source_types = set(s.source_type for s in sources)
        if len(source_types) >= self.MIN_SOURCE_TYPES:
            passed.append(f"Source diversity ({len(source_types)} types)")
        else:
            failed.append(f"Low source diversity ({len(source_types)} < {self.MIN_SOURCE_TYPES} types)")

        # Gate 3: Claim support rate
        if claim_validations:
            support_rate = sum(1 for c in claim_validations if c.is_supported) / len(claim_validations)
            if support_rate >= self.MIN_CLAIM_SUPPORT:
                passed.append(f"Claim support rate ({support_rate:.0%} >= {self.MIN_CLAIM_SUPPORT:.0%})")
            else:
                failed.append(f"Low claim support ({support_rate:.0%} < {self.MIN_CLAIM_SUPPORT:.0%})")

        # Gate 4: Source consistency
        if consistency.consistency_score >= self.MIN_CONSISTENCY:
            passed.append(f"Source consistency ({consistency.consistency_score:.0%})")
        else:
            failed.append(f"Source inconsistency ({consistency.consistency_score:.0%} < {self.MIN_CONSISTENCY:.0%})")

        # Gate 5: Minimum confidence
        if confidence.score >= self.MIN_CONFIDENCE:
            passed.append(f"Confidence threshold ({confidence.score:.0%})")
        else:
            failed.append(f"Low confidence ({confidence.score:.0%} < {self.MIN_CONFIDENCE:.0%})")

        # Gate 6: No critical issues (strong claims without support)
        unsupported_strong = [
            c for c in claim_validations
            if not c.is_supported and any(m in c.claim.lower() for m in self.STRONG_CLAIM_MARKERS)
        ]
        if not unsupported_strong:
            passed.append("No unsupported strong claims")
        else:
            failed.append(f"{len(unsupported_strong)} unsupported strong claim(s)")

        return passed, failed


def quick_validate(synthesis: dict, findings: list[dict], sources: list,
                   strict: bool = False) -> ValidationResult:
    """
    Quick validation of research output.

    Args:
        synthesis: The synthesized output
        findings: List of {query, source, finding} dicts
        sources: List of Source objects
        strict: Use stricter thresholds

    Returns:
        ValidationResult

    Example:
        from researcher.validator import quick_validate

        result = quick_validate(synthesis, findings, sources)
        if not result.is_valid:
            print("Research failed quality gates:")
            for gate in result.failed_gates:
                print(f"  - {gate}")
    """
    validator = ResearchValidator(strict_mode=strict)
    return validator.validate(synthesis, findings, sources)
