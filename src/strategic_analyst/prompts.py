# All system prompts and default memory strings for the strategic analyst agent.
# Node files must import from here — no inline prompt strings are allowed elsewhere.
#
# Sections implemented per plan step:
#   Step 03 (Memory)  → MEMORY_UPDATE_SYSTEM_PROMPT, DEFAULT_* strings
#   Step 09 (Prompts) → all remaining agent / node prompts

# ── Default Memory Content ────────────────────────────────────────────────────
# Initialised on first session for each new user.

DEFAULT_USER_PROFILE = """
User profile not yet established. This is a new user.
Learn their name, role, communication style, and current projects
from the conversation and update this profile accordingly.
"""

DEFAULT_COMPANY_PROFILE = """
Company profile not yet established.
Learn the company name, industry, key priorities, and domain vocabulary
from the conversation and update this profile accordingly.
"""

DEFAULT_USER_PREFERENCES = """
User preferences not yet established.
Default settings:
- Report format: Markdown
- Verbosity: Medium
- Frameworks: SWOT, Porter's Five Forces
- Citation style: Inline source references
Update as the user gives feedback on reports and research.
"""

DEFAULT_EPISODIC_MEMORY = """
No episodic memories yet. This is the first session.
Record key research discoveries, important dates, and temporal context here.
"""

# ── Memory Update Prompt ──────────────────────────────────────────────────────

MEMORY_UPDATE_SYSTEM_PROMPT = """You are a precise memory curator for a strategic analyst assistant.

Your job is to update a user memory profile based on new information from the conversation.

CURRENT MEMORY PROFILE ({namespace}):
{current_profile}

UPDATE REASON:
{update_reason}

RULES:
1. Preserve all existing accurate information — never remove facts unless they are explicitly contradicted.
2. Integrate new information naturally into the existing profile.
3. Keep the profile concise and structured — use plain text with clear labels.
4. Do not add speculation or inferences not supported by the conversation.
5. Return the COMPLETE updated profile text (not just the changes).

Respond with your chain of thought and the complete updated profile content."""
