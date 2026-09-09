"""Common task-handling policy shared by language-model systems."""

TASK_POLICY_VERSION = 1
TASK_POLICY = (
    "Task policy. The structured goal list takes precedence over a physician note. "
    "If the note conflicts with that list, either escalate with reason contradictory_instructions "
    "and identify the conflict, or plan to the structured goals and explicitly identify the "
    "conflicting requirements in the submission note. Do not silently follow a conflicting note. "
    "A benign note is not a reason to escalate. A goal naming an absent structure requires "
    "escalation with reason missing_structure. Use infeasible for a demonstrated incompatibility "
    "among hard dose requirements, unit_ambiguity for unresolved dose units, or other for a "
    "different problem that prevents planning. One optimization that misses a goal is not proof "
    "of infeasibility. When no such problem is established, plan within the available budget "
    "and record unmet goals honestly."
)
