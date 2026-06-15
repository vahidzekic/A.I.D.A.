"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: parser.py
PHASE: 4 (Symbolic Engine — Bulletproof Output Parsing)
PURPOSE: Safely extract and validate JSON tool calls from raw LLM output,
         no matter how malformed, wrapped, or garbled it might be.

WHY IS THIS NECESSARY?
    LLMs — especially small "Baby" ones — are notoriously unreliable at
    producing perfectly formatted JSON. Common failure modes include:

    1. Wrapping JSON in markdown code blocks:
       ```json
       {"tool": "calculator", "args": {"expression": "2+3"}}
       ```

    2. Adding conversational fluff around the JSON:
       "Sure! Let me calculate that for you.
        {"tool": "calculator", "args": {"expression": "2+3"}}
        Hope that helps!"

    3. Malformed JSON:
       {"tool": "calculator", "args": {"expression": "2+3",}}  ← trailing comma
       {'tool': 'calculator'}   ← single quotes instead of double
       {tool: calculator}       ← missing quotes entirely

    4. Truncated output (model hit max_tokens mid-JSON):
       {"tool": "calculator", "args": {"express

    This parser handles ALL of these cases gracefully. If parsing fails,
    it returns a structured error that can be fed BACK to the LLM in the
    ReAct loop, giving it a chance to correct itself.

PARSING STRATEGY (Multi-Layer):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Layer 1: Check for <|tool_call|> special token                  │
    │           → If found, extract everything after it               │
    │                                                                  │
    │  Layer 2: Try to extract JSON from markdown code blocks          │
    │           → Regex: find code-fenced JSON blocks                │
    │                                                                  │
    │  Layer 3: Try to find raw JSON object in the text                │
    │           → Regex: find matching { ... } with brace counting    │
    │                                                                  │
    │  Layer 4: Try json.loads() on the extracted string               │
    │           → If it works, validate the tool call schema           │
    │                                                                  │
    │  Layer 5: If json.loads() fails, attempt repair:                 │
    │           → Fix trailing commas                                  │
    │           → Fix single quotes → double quotes                   │
    │           → Strip control characters                             │
    │           → Retry json.loads()                                   │
    │                                                                  │
    │  Layer 6: If ALL fails, return structured error for ReAct retry  │
    └──────────────────────────────────────────────────────────────────┘

=============================================================================
"""

import re
import json
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# 1. PARSING RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════
class ParseResult:
    """
    Structured result of parsing LLM output.

    Attributes:
        is_tool_call:  True if the output contains a valid tool call
        tool_name:     Name of the tool (if is_tool_call)
        tool_args:     Arguments dict (if is_tool_call)
        raw_text:      The original LLM output (for direct text responses)
        error_message: Description of what went wrong (if parsing failed)
        status:        "tool_call" | "text_response" | "error"
    """

    def __init__(
        self,
        status: str,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        raw_text: str = "",
        error_message: str | None = None,
    ):
        self.status = status
        self.is_tool_call = (status == "tool_call")
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.raw_text = raw_text
        self.error_message = error_message

    def to_dict(self) -> dict:
        """Convert to a dictionary (for logging and debugging)."""
        d = {"status": self.status}
        if self.is_tool_call:
            d["tool_name"] = self.tool_name
            d["tool_args"] = self.tool_args
        elif self.status == "error":
            d["error_message"] = self.error_message
            d["raw_text"] = self.raw_text[:200]  # Truncate for readability
        else:
            d["text"] = self.raw_text
        return d

    def __repr__(self) -> str:
        return f"ParseResult({json.dumps(self.to_dict(), indent=2)})"


# ═══════════════════════════════════════════════════════════════════════════
# 2. THE MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════════
def parse_llm_output(text: str) -> ParseResult:
    """
    Parse raw LLM output into a structured result.

    This is the PRIMARY INTERFACE used by the Agentic Loop (Phase 5).

    The parser determines if the LLM wants to:
    1. Call a tool  → returns ParseResult(status="tool_call", ...)
    2. Respond with text → returns ParseResult(status="text_response", ...)
    3. (Error case) → returns ParseResult(status="error", ...)

    Args:
        text: Raw string output from the LLM.

    Returns:
        ParseResult with the parsed information.
    """
    if not text or not text.strip():
        return ParseResult(
            status="text_response",
            raw_text="",
        )

    text = text.strip()

    # ── Layer 1: Check for <|tool_call|> special token ─────────────────
    # If the LLM emitted <|tool_call|>, it INTENDS to call a tool.
    # Extract everything after the token as the JSON payload.
    tool_call_marker = "<|tool_call|>"
    if tool_call_marker in text:
        # Extract the part after <|tool_call|>
        json_part = text.split(tool_call_marker, 1)[1].strip()

        # Also strip any trailing <|eos|> or similar tokens
        json_part = _strip_special_tokens(json_part)

        # Try to parse the JSON
        result = _try_parse_json(json_part)
        if result is not None:
            return _validate_tool_call(result, text)

        # If direct parse failed, try extracting JSON from the mess
        extracted = _extract_json_string(json_part)
        if extracted:
            result = _try_parse_json(extracted)
            if result is not None:
                return _validate_tool_call(result, text)

        # Last resort: try to repair the JSON
        repaired = _try_repair_json(json_part)
        if repaired is not None:
            return _validate_tool_call(repaired, text)

        # Tool call was intended but JSON is unrecoverable
        return ParseResult(
            status="error",
            raw_text=text,
            error_message=(
                f"You emitted <|tool_call|> but the JSON that followed "
                f"is malformed and could not be parsed. "
                f"The broken JSON was: '{json_part[:150]}'. "
                f"Please respond with ONLY valid JSON in this exact format: "
                f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}'
            ),
        )

    # ── Layer 2: No <|tool_call|> token — check for JSON anyway ────────
    # Some models might output raw JSON without the special token.
    extracted = _extract_json_string(text)
    if extracted:
        result = _try_parse_json(extracted)
        if result is not None:
            # Check if it looks like a tool call
            if isinstance(result, dict) and "tool" in result:
                return _validate_tool_call(result, text)

    # ── Layer 3: It's a plain text response ────────────────────────────
    # The LLM is responding conversationally (no tool call).
    # Strip any remaining special tokens for clean output.
    clean_text = _strip_special_tokens(text)
    return ParseResult(
        status="text_response",
        raw_text=clean_text,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. INTERNAL HELPERS — JSON Extraction & Repair
# ═══════════════════════════════════════════════════════════════════════════

def _strip_special_tokens(text: str) -> str:
    """Remove A.I.D.A. special tokens from text."""
    special_tokens = [
        "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
        "<|tool_call|>", "<|tool_result|>",
        "<|user|>", "<|assistant|>", "<|system|>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


def _extract_json_string(text: str) -> str | None:
    """
    Extract a JSON object string from potentially messy text.

    Strategy (tried in order):
    1. Look for markdown code blocks: ```json ... ``` or ``` ... ```
    2. Look for the first { ... } with balanced brace counting
    """
    # ── Strategy 1: Markdown code blocks ───────────────────────────────
    # Match ```json ... ``` or ``` ... ```
    # re.DOTALL makes . match newlines too
    md_patterns = [
        r"```json\s*(.*?)\s*```",   # ```json ... ```
        r"```\s*(.*?)\s*```",       # ``` ... ```
    ]

    for pattern in md_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted.startswith("{"):
                return extracted

    # ── Strategy 2: Balanced brace matching ────────────────────────────
    # Find the first '{' and match it to its closing '}'
    # This handles cases like: "Sure, here you go: {"tool": "calc"} enjoy!"
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]

    # Unbalanced braces — return what we found anyway (repair might fix it)
    if start_idx < len(text):
        return text[start_idx:]

    return None


def _try_parse_json(text: str) -> dict | None:
    """
    Attempt to parse a string as JSON.

    Returns the parsed dict if successful, None otherwise.
    NEVER raises an exception.
    """
    if not text:
        return None

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _try_repair_json(text: str) -> dict | None:
    """
    Attempt to fix common JSON errors produced by LLMs.

    Repairs tried (in order):
    1. Remove trailing commas before } or ]
    2. Replace single quotes with double quotes
    3. Remove control characters
    4. Add missing closing braces
    5. Strip trailing garbage after the last }

    Returns the parsed dict if repair succeeds, None otherwise.
    NEVER raises an exception.
    """
    if not text:
        return None

    repaired = text

    # ── Repair 1: Remove trailing commas ───────────────────────────────
    # {"a": 1, "b": 2,}  →  {"a": 1, "b": 2}
    # This is the #1 most common LLM JSON error.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # ── Repair 2: Single quotes → double quotes ───────────────────────
    # {'tool': 'calc'}  →  {"tool": "calc"}
    # Careful: don't replace apostrophes inside words.
    # We use a simple heuristic: if a quote is adjacent to { } [ ] , :
    # it's likely a JSON quote, not an apostrophe.
    repaired = re.sub(r"(?<=[\[{,:\s])'|'(?=[\]}:,\s])", '"', repaired)

    # ── Repair 3: Remove control characters ────────────────────────────
    # \x00-\x1f except \n \r \t (which are valid in JSON strings)
    repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", repaired)

    # ── Repair 4: Try to close unclosed braces ─────────────────────────
    open_braces = repaired.count("{") - repaired.count("}")
    if open_braces > 0:
        repaired += "}" * open_braces

    open_brackets = repaired.count("[") - repaired.count("]")
    if open_brackets > 0:
        repaired += "]" * open_brackets

    # ── Repair 5: Strip garbage after last } ───────────────────────────
    last_brace = repaired.rfind("}")
    if last_brace != -1:
        repaired = repaired[:last_brace + 1]

    # ── Attempt to parse the repaired JSON ─────────────────────────────
    result = _try_parse_json(repaired)
    if result is not None:
        return result

    # ── Final attempt: extract and retry ───────────────────────────────
    extracted = _extract_json_string(repaired)
    if extracted and extracted != repaired:
        return _try_parse_json(extracted)

    return None


def _validate_tool_call(parsed: dict, original_text: str) -> ParseResult:
    """
    Validate that a parsed JSON dict is a proper tool call.

    Expected format:
        {"tool": "tool_name", "args": {"arg1": "val1", ...}}

    Also accepts:
        {"tool": "tool_name", "args": {"arg1": "val1"}}
        {"tool": "tool_name"}  (no args = empty args)
    """
    # ── Check for "tool" key ───────────────────────────────────────────
    if "tool" not in parsed:
        return ParseResult(
            status="error",
            raw_text=original_text,
            error_message=(
                f"The JSON was parsed successfully, but it's missing the "
                f"required 'tool' key. Got keys: {list(parsed.keys())}. "
                f"Expected format: "
                f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}'
            ),
        )

    tool_name = parsed["tool"]
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ParseResult(
            status="error",
            raw_text=original_text,
            error_message=(
                f"The 'tool' value must be a non-empty string, "
                f"but got: {repr(tool_name)}"
            ),
        )

    # ── Extract args (default to empty dict) ───────────────────────────
    tool_args = parsed.get("args", {})
    if not isinstance(tool_args, dict):
        # Try to recover: maybe args is a string that's actually JSON
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        else:
            tool_args = {}

    return ParseResult(
        status="tool_call",
        tool_name=tool_name.strip(),
        tool_args=tool_args,
        raw_text=original_text,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Test the parser with various inputs
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — Symbolic Engine: Parser Verification")
    print("=" * 70)

    test_cases = [
        # ── Test 1: Perfect JSON tool call with <|tool_call|> ──────────
        {
            "name": "Perfect JSON with <|tool_call|> token",
            "input": '<|tool_call|>{"tool": "calculator", "args": {"expression": "15 + 27"}}',
            "expect_status": "tool_call",
            "expect_tool": "calculator",
        },

        # ── Test 2: Tool call hidden in conversational text ────────────
        {
            "name": "JSON buried in conversational text",
            "input": (
                "Sure, I'll help you calculate that! "
                '<|tool_call|>{"tool": "get_balance", "args": {"username": "john"}} '
                "Let me know if you need anything else!"
            ),
            "expect_status": "tool_call",
            "expect_tool": "get_balance",
        },

        # ── Test 3: Horribly broken/malformed JSON ─────────────────────
        {
            "name": "Completely malformed JSON (trailing comma + single quotes)",
            "input": "<|tool_call|>{'tool': 'calculator', 'args': {'expression': '2 + 3',}}",
            "expect_status": "tool_call",  # Should be REPAIRED
            "expect_tool": "calculator",
        },

        # ── Test 4: JSON in markdown code block ────────────────────────
        {
            "name": "JSON wrapped in markdown code block",
            "input": (
                'Here is the tool call:\n'
                '```json\n'
                '{"tool": "calculator", "args": {"expression": "10 * 5"}}\n'
                '```\n'
            ),
            "expect_status": "tool_call",  # JSON with "tool" key is detected
            "expect_tool": "calculator",
            # Even without <|tool_call|>, raw JSON with "tool" key is caught
        },

        # ── Test 5: Markdown with <|tool_call|> ────────────────────────
        {
            "name": "Markdown code block after <|tool_call|>",
            "input": (
                '<|tool_call|>```json\n'
                '{"tool": "get_balance", "args": {"username": "alice"}}\n'
                '```'
            ),
            "expect_status": "tool_call",
            "expect_tool": "get_balance",
        },

        # ── Test 6: Truncated JSON (model hit max_tokens) ─────────────
        {
            "name": "Truncated JSON (unclosed braces)",
            "input": '<|tool_call|>{"tool": "calculator", "args": {"expression": "2+3"',
            "expect_status": "tool_call",  # Repair should close the braces
            "expect_tool": "calculator",
        },

        # ── Test 7: Completely garbled output ──────────────────────────
        {
            "name": "Totally garbled (unrepairable)",
            "input": "<|tool_call|>asdfghjkl this is not json at all {{{{{",
            "expect_status": "error",
            "expect_tool": None,
        },

        # ── Test 8: Plain text response (no tool call) ─────────────────
        {
            "name": "Plain text response (greeting)",
            "input": "Hello! How can I help you today?",
            "expect_status": "text_response",
            "expect_tool": None,
        },

        # ── Test 9: Tool call without args ─────────────────────────────
        {
            "name": "Tool call with no args",
            "input": '<|tool_call|>{"tool": "get_balance"}',
            "expect_status": "tool_call",
            "expect_tool": "get_balance",
        },

        # ── Test 10: Missing "tool" key ────────────────────────────────
        {
            "name": "JSON with missing 'tool' key",
            "input": '<|tool_call|>{"action": "calculate", "data": "2+3"}',
            "expect_status": "error",  # Valid JSON but wrong schema
            "expect_tool": None,
        },

        # ── Test 11: Extra whitespace and newlines ─────────────────────
        {
            "name": "Extra whitespace and newlines",
            "input": (
                '  <|tool_call|>  \n\n  '
                '{"tool": "calculator",\n'
                '  "args": {\n'
                '    "expression": "100 / 4"\n'
                '  }\n'
                '}\n  '
            ),
            "expect_status": "tool_call",
            "expect_tool": "calculator",
        },

        # ── Test 12: Response with <|eos|> at the end ──────────────────
        {
            "name": "Tool call followed by <|eos|>",
            "input": '<|tool_call|>{"tool": "calculator", "args": {"expression": "7 * 8"}}<|eos|>',
            "expect_status": "tool_call",
            "expect_tool": "calculator",
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"  Test {i:2d}: {test['name']}")
        print(f"{'─' * 70}")
        print(f"  Input:  {test['input'][:80]}{'...' if len(test['input']) > 80 else ''}")

        result = parse_llm_output(test["input"])

        print(f"  Status: {result.status}")
        if result.is_tool_call:
            print(f"  Tool:   {result.tool_name}")
            print(f"  Args:   {json.dumps(result.tool_args)}")
        elif result.status == "error":
            print(f"  Error:  {result.error_message[:80]}...")
        else:
            print(f"  Text:   {result.raw_text[:80]}")

        # Check expectations
        status_ok = result.status == test["expect_status"]
        tool_ok = result.tool_name == test["expect_tool"]

        if status_ok and tool_ok:
            print(f"  Result: ✅ PASSED")
            passed += 1
        else:
            print(f"  Result: ❌ FAILED")
            if not status_ok:
                print(f"          Expected status: {test['expect_status']}, "
                      f"got: {result.status}")
            if not tool_ok:
                print(f"          Expected tool: {test['expect_tool']}, "
                      f"got: {result.tool_name}")
            failed += 1

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Parser Verification Summary:")
    print(f"  Passed: {passed}/{passed + failed}")
    print(f"  Failed: {failed}/{passed + failed}")

    # THE CRITICAL ASSERTION:
    # The script MUST NOT crash on any input, no matter how garbled.
    print(f"\n  🛡️  CRITICAL: Script completed without crashing!")
    print(f"     (Even with malformed JSON, truncated output, and garbled text)")

    if failed == 0:
        print(f"\n  ✅ All parser tests passed!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed — review above.")

    print("=" * 70)
