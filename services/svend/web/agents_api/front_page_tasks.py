"""Front page digest generation — async Qwen 14B via Tempora.

Collects all learnings (front matter, Hansei Kai, yokoten) for a user,
runs Qwen 2.5-Coder-14B to extract themes and contradictions, caches
the result in FrontPageDigest.

Triggered on notebook conclusion or front matter addition.
"""

import hashlib
import json
import logging
import time

from django.contrib.auth import get_user_model
from django.utils import timezone

logger = logging.getLogger("svend.front_page")
User = get_user_model()

# The prompt template for theme extraction
_PROMPT = """You are analyzing learnings from continuous improvement notebooks. Be concise and insightful.

LEARNINGS:
{learnings}

Respond in EXACTLY this JSON format, no other text:
{{
  "themes": [
    {{"name": "theme name", "items": [0, 3, 7], "summary": "one sentence"}}
  ],
  "contradictions": [
    {{"items": [0, 4], "note": "one sentence"}}
  ],
  "digest": "one paragraph, max 3 sentences"
}}"""


def _collect_learnings(user):
    """Collect all learning strings for a user across notebooks."""
    from core.models import HanseiKai, NotebookPage, Yokoten

    items = []

    # Hansei Kai key learnings
    for hk in HanseiKai.objects.filter(notebook__owner=user).order_by("-created_at"):
        items.append(f"[Reflection] {hk.key_learning}")

    # Anti-patterns and notes from front matter
    for page in NotebookPage.objects.filter(notebook__owner=user, trial_role="front_matter").order_by("-created_at"):
        label = "Anti-pattern" if page.source_tool == "anti_pattern" else "Note"
        text = page.narrative or page.title
        items.append(f"[{label}] {text}")

    # Yokoten
    for y in Yokoten.objects.filter(source_notebook__owner=user).order_by("-created_at"):
        ctx = f" ({y.context})" if y.context else ""
        items.append(f"[Yokoten] {y.learning}{ctx}")

    return items


def _compute_hash(items):
    """Hash the learning items to detect changes."""
    content = "\n".join(items)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _run_qwen(learnings_text):
    """Run Qwen 2.5-Coder-14B inference for theme extraction."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
    logger.info("Loading %s for front page digest...", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    prompt = _PROMPT.format(learnings=learnings_text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )
    gen_time = time.time() - start

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    tokens_out = outputs.shape[1] - inputs["input_ids"].shape[1]

    # Cleanup GPU memory
    del model, inputs, outputs
    torch.cuda.empty_cache()

    logger.info("Qwen generated %d tokens in %.1fs (%.0f tok/s)", tokens_out, gen_time, tokens_out / gen_time)

    return response, gen_time * 1000


def _parse_response(response_text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response_text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return {
            "themes": data.get("themes", []),
            "contradictions": data.get("contradictions", []),
            "digest": data.get("digest", ""),
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse Qwen response as JSON: %s", text[:200])
        return {"themes": [], "contradictions": [], "digest": ""}


def generate_front_page_digest(payload, context=None):
    """Tempora task handler: generate front page digest for a user.

    payload: {"user_id": int}
    """
    from core.models import FrontPageDigest

    user_id = payload.get("user_id")
    if not user_id:
        return {"error": "missing user_id"}

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return {"error": "user not found"}

    # Collect learnings
    items = _collect_learnings(user)
    if not items:
        logger.info("No learnings for user %s — skipping digest", user.email)
        return {"skipped": True, "reason": "no_learnings"}

    # Check if regeneration is needed
    source_hash = _compute_hash(items)
    existing = FrontPageDigest.objects.filter(user=user).first()
    if existing and existing.source_hash == source_hash:
        logger.info("Digest for %s is current (hash=%s) — skipping", user.email, source_hash)
        return {"skipped": True, "reason": "hash_unchanged"}

    # Format learnings as numbered list
    learnings_text = "\n".join(f"- {item}" for item in items)

    # Run LLM
    try:
        response_text, gen_time_ms = _run_qwen(learnings_text)
        parsed = _parse_response(response_text)
    except Exception as e:
        logger.exception("Qwen inference failed for user %s", user.email)
        return {"error": str(e)}

    # Save or update digest
    digest, _ = FrontPageDigest.objects.update_or_create(
        user=user,
        defaults={
            "themes": parsed["themes"],
            "contradictions": parsed["contradictions"],
            "digest": parsed["digest"],
            "source_hash": source_hash,
            "source_items": items,
            "generated_at": timezone.now(),
            "generation_time_ms": gen_time_ms,
        },
    )

    logger.info(
        "Front page digest generated for %s: %d themes, %d contradictions (%.1fs)",
        user.email,
        len(parsed["themes"]),
        len(parsed["contradictions"]),
        gen_time_ms / 1000,
    )

    return {
        "user": user.email,
        "themes": len(parsed["themes"]),
        "contradictions": len(parsed["contradictions"]),
        "generation_time_ms": gen_time_ms,
    }
