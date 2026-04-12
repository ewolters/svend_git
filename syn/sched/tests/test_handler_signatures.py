"""
Regression test: all registered task handlers must accept (payload, context).

The executor calls handler(payload, context). Any handler with a different
signature will raise TypeError at runtime. This test catches that at test
time by inspecting every registered handler's signature.

Ref: CR-84be0f1c — compliance_daily and 9 other handlers had wrong signature,
causing 13 stuck RUNNING tasks and 12 DLQ entries over 12 days.
"""

import inspect

from django.test import SimpleTestCase

from syn.sched.core import TaskRegistry


class HandlerSignatureTest(SimpleTestCase):
    """Every registered handler must accept (payload, context) — SCH-001."""

    def test_all_handlers_accept_payload_and_context(self):
        """Handlers called as handler(payload, context) must not raise TypeError.

        The executor passes exactly 2 positional args: payload (dict) and
        context (ExecutionContext). Handlers that accept fewer will crash
        at runtime with 'takes N positional argument(s) but 2 were given'.
        """
        # Force registration (may already be done at import time)
        from syn.sched.svend_tasks import register_svend_tasks

        try:
            register_svend_tasks()
        except Exception:
            pass  # Already registered

        handlers = TaskRegistry._handlers
        self.assertGreater(len(handlers), 0, "No handlers registered")

        bad = []
        for name, handler in handlers.items():
            sig = inspect.signature(handler)
            params = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            # Must accept at least 2 positional args (payload, context)
            # or use *args / **kwargs
            has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
            has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

            total_positional = len(
                [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            )

            if has_var_positional or has_var_keyword:
                continue  # *args or **kwargs — accepts anything

            if total_positional < 2:
                bad.append(f"{name}: {handler.__qualname__}{sig} — needs (payload, context)")

        if bad:
            self.fail(
                f"{len(bad)} handler(s) with wrong signature (must accept 2 args: payload, context):\n"
                + "\n".join(f"  - {b}" for b in bad)
            )
