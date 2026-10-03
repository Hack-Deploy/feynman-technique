"""Settlement: the only module that calls the oracle.

A venue produces a ``SubmittedAttempt``; ``settle`` scores its law against the
preregistration fixed when the prize was posted and returns the final
``AttemptRecord`` with the verdict. Markets get preregistrations through
``prereg_for`` so they never import the oracle either.
"""

from __future__ import annotations

import hashlib
import uuid

from dm import oracle
from dm.types import AttemptRecord, Preregistration, SubmittedAttempt


def prereg_for(venue: str, world: str, test_seed: int) -> Preregistration:
    return oracle.make_prereg(venue, world, test_seed)


def attempt_id_for(s: SubmittedAttempt, commitment: str) -> str:
    law_hash = hashlib.sha256((s.submitted_law or "").encode()).hexdigest()[:16]
    key = f"{s.source}:{s.protocol}:{s.venue}:{s.world}:{s.solver}:{s.seed}:{commitment}:{law_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def settle(prereg: Preregistration, submitted: SubmittedAttempt,
           timeout_s: float = oracle.DEFAULT_TIMEOUT_S,
           with_explanation: bool = False) -> AttemptRecord:
    if (prereg.venue, prereg.world) != (submitted.venue, submitted.world):
        raise ValueError(
            f"preregistration is for {prereg.question_id}, "
            f"attempt is for {submitted.venue}/{submitted.world}")
    verdict = oracle.score(prereg, submitted.submitted_law,
                           training=submitted.training, timeout_s=timeout_s)
    if with_explanation:
        verdict["explanation_score"] = oracle.explain_score(
            submitted.world, submitted.explanation)
    return AttemptRecord(
        attempt_id=attempt_id_for(submitted, prereg.commitment()),
        source=submitted.source,
        protocol=submitted.protocol,
        venue=submitted.venue,
        world=submitted.world,
        solver=submitted.solver,
        seed=submitted.seed,
        stated_p_success=submitted.stated_p_success,
        rounds=submitted.rounds,
        experiments=submitted.experiments,
        lab_cost=submitted.lab_cost,
        llm_usage=submitted.llm_usage,
        submitted_law=submitted.submitted_law,
        verdict=verdict,
        transcript_path=submitted.transcript_path,
        created_at=submitted.created_at,
        extra={**submitted.extra, "explanation": submitted.explanation},
    )
