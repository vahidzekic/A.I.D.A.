"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: tools.py
PHASE: 4 (Symbolic Engine — Deterministic Logic)
PURPOSE: The "hands" of A.I.D.A. — concrete Python functions that the LLM
         can invoke to perform exact computation, database lookups, and
         other deterministic operations.

WHY "SYMBOLIC"?
    The LLM (Phase 1-3) is NEURAL — it operates on probabilities, fuzzy
    pattern matching, and learned representations. It's great at language
    understanding but TERRIBLE at exact math, database queries, and
    anything requiring deterministic precision.

    The Symbolic Engine is the opposite — no learning, no probability.
    Just precise, deterministic Python functions that always produce
    the exact correct answer.

    The Agentic Loop (Phase 5) bridges them:
        USER → LLM decides WHICH tool → Symbolic Engine executes → LLM speaks

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Tool (Abstract Base Class)                                      │
    │    ├── name: str          — unique identifier for the tool       │
    │    ├── description: str   — what the tool does (for the LLM)     │
    │    ├── parameters: dict   — schema of required arguments         │
    │    └── execute(**kwargs)  — the actual deterministic logic        │
    │                                                                  │
    │  ToolRegistry                                                    │
    │    ├── register(tool)     — add a tool to the registry           │
    │    ├── get_tool(name)     — look up a tool by name               │
    │    ├── get_schema()       — return JSON schema for system prompt │
    │    └── execute(name, **kwargs) — find + run a tool               │
    │                                                                  │
    │  Concrete Tools:                                                 │
    │    ├── CalculatorTool     — safe math evaluation (NO eval()!)    │
    │    └── AccountBalanceTool — mock database lookup                 │
    └──────────────────────────────────────────────────────────────────┘

=============================================================================
"""

import ast
import operator
import json
from abc import ABC, abstractmethod
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOOL BASE CLASS — Abstract interface for all tools
# ═══════════════════════════════════════════════════════════════════════════
class Tool(ABC):
    """
    Abstract base class for all deterministic tools.

    Every tool MUST define:
        - name:        A unique string identifier (used in JSON tool calls)
        - description: A human-readable description (injected into LLM prompt)
        - parameters:  A dict describing required arguments and their types
        - execute():   The actual computation logic

    THE TOOL CONTRACT:
        1. Tools are DETERMINISTIC — same input always produces same output.
        2. Tools NEVER raise unhandled exceptions — they return error dicts.
        3. Tools return a dict with at least {"status": "success"/"error"}.
        4. The parameter schema tells the LLM exactly what args to provide.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool (e.g. 'calculator')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this tool does, in plain English. The LLM reads this."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        Schema of required arguments.

        Format:
        {
            "arg_name": {
                "type": "string" | "number" | "integer",
                "description": "What this argument is for"
            },
            ...
        }
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: The arguments matching the parameter schema.

        Returns:
            A dict with at least:
                {"status": "success", "result": <value>}
            or:
                {"status": "error", "error_message": "<description>"}
        """
        ...

    def get_schema(self) -> dict:
        """
        Return this tool's full schema as a dictionary.

        This schema is injected into the LLM's system prompt so it
        knows what tools are available and how to call them.

        Example output:
        {
            "name": "calculator",
            "description": "Safely evaluates a mathematical expression...",
            "parameters": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '2 + 3 * 4'"
                }
            }
        }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. TOOL REGISTRY — Central catalog of all available tools
# ═══════════════════════════════════════════════════════════════════════════
class ToolRegistry:
    """
    Central registry that holds all available tools.

    The registry serves three purposes:
    1. DISCOVERY — The LLM needs to know what tools exist (via schema).
    2. DISPATCH — Given a tool name and args, find and execute the tool.
    3. SAFETY  — Reject calls to unknown tools with a clear error.

    Usage:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(AccountBalanceTool())

        # Get schema for system prompt
        schema = registry.get_schema()

        # Execute a tool call from the LLM
        result = registry.execute("calculator", expression="2 + 3")
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: An instance of a Tool subclass.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Each tool must have a unique name."
            )
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def get_schema(self) -> list[dict]:
        """
        Return the complete schema for ALL registered tools.

        This is what gets injected into the LLM's system prompt.
        The LLM reads this schema to understand:
            - What tools are available
            - What arguments each tool requires
            - What each tool does

        Returns:
            A list of tool schema dictionaries.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def get_schema_string(self) -> str:
        """
        Return the schema as a formatted JSON string.

        This is the exact text that gets injected into the system prompt.
        Pretty-printed for readability (the LLM can handle it).
        """
        return json.dumps(self.get_schema(), indent=2)

    def execute(self, tool_name: str, **kwargs) -> dict:
        """
        Look up and execute a tool by name.

        This is the main entry point called by the Agentic Loop (Phase 5).

        Args:
            tool_name: The name of the tool to execute.
            **kwargs:  Arguments to pass to the tool.

        Returns:
            Tool result dict with "status" field, OR an error dict if
            the tool is not found or execution fails.
        """
        # ── Validate tool exists ───────────────────────────────────────
        tool = self.get_tool(tool_name)
        if tool is None:
            available = ", ".join(self.list_tools()) or "(none)"
            return {
                "status": "error",
                "error_message": (
                    f"Unknown tool: '{tool_name}'. "
                    f"Available tools: [{available}]"
                ),
            }

        # ── Validate required parameters ───────────────────────────────
        required_params = set(tool.parameters.keys())
        provided_params = set(kwargs.keys())
        missing = required_params - provided_params

        if missing:
            return {
                "status": "error",
                "error_message": (
                    f"Missing required arguments for tool '{tool_name}': "
                    f"{sorted(missing)}. "
                    f"Required: {sorted(required_params)}"
                ),
            }

        # ── Execute with safety net ────────────────────────────────────
        try:
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            return {
                "status": "error",
                "error_message": (
                    f"Tool '{tool_name}' raised an exception: "
                    f"{type(e).__name__}: {str(e)}"
                ),
            }


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONCRETE TOOL: Safe Calculator
# ═══════════════════════════════════════════════════════════════════════════
class CalculatorTool(Tool):
    """
    Safely evaluates basic mathematical expressions.

    ┌──────────────────────────────────────────────────────────────────┐
    │  WHY NOT JUST USE eval()?                                        │
    │                                                                  │
    │  eval("2 + 3")  →  5        ✅ Works!                           │
    │  eval("__import__('os').system('rm -rf /')") → DISASTER 💀      │
    │                                                                  │
    │  eval() executes ARBITRARY Python code. If the LLM generates    │
    │  a malicious string (or is tricked via prompt injection), eval() │
    │  could delete files, exfiltrate data, or worse.                  │
    │                                                                  │
    │  OUR APPROACH: Parse the expression into an AST (Abstract Syntax │
    │  Tree) and only allow arithmetic operations on numbers.          │
    │  Any attempt to call functions, access attributes, or import     │
    │  modules is REJECTED at the AST level.                           │
    └──────────────────────────────────────────────────────────────────┘

    Supported Operations:
        + (addition), - (subtraction), * (multiplication),
        / (division), ** (power), // (floor division), % (modulo)
        Unary + and - (e.g., -5, +3)
        Parentheses for grouping

    Examples:
        "2 + 3"         → 5
        "10 * (3 + 2)"  → 50
        "2 ** 10"       → 1024
        "17 % 5"        → 2
        "10 / 3"        → 3.3333...
    """

    # Map AST operator nodes to actual Python operator functions.
    # This is the WHITELIST — only these operations are allowed.
    _ALLOWED_BINARY_OPS = {
        ast.Add:      operator.add,       # +
        ast.Sub:      operator.sub,       # -
        ast.Mult:     operator.mul,       # *
        ast.Div:      operator.truediv,   # /
        ast.FloorDiv: operator.floordiv,  # //
        ast.Mod:      operator.mod,       # %
        ast.Pow:      operator.pow,       # **
    }

    _ALLOWED_UNARY_OPS = {
        ast.UAdd: operator.pos,  # +x
        ast.USub: operator.neg,  # -x
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Safely evaluates a mathematical expression. "
            "Supports: +, -, *, /, //, %, ** and parentheses. "
            "Example: '15 + 27' returns 42."
        )

    @property
    def parameters(self) -> dict:
        return {
            "expression": {
                "type": "string",
                "description": (
                    "A mathematical expression to evaluate. "
                    "Example: '2 + 3 * 4' or '(10 - 3) ** 2'"
                ),
            }
        }

    def _safe_eval_node(self, node: ast.AST) -> float | int:
        """
        Recursively evaluate an AST node, allowing ONLY arithmetic.

        This walks the parsed expression tree node by node:
        - Numbers (ast.Constant) → return the literal value
        - Binary ops (ast.BinOp) → evaluate left and right, apply operator
        - Unary ops (ast.UnaryOp) → evaluate operand, apply operator
        - ANYTHING ELSE → raise ValueError (blocks function calls, etc.)

        This is the SECURITY BOUNDARY. No function calls, no attribute
        access, no imports — just pure arithmetic on literal numbers.
        """
        # ── Literal number (int or float) ──────────────────────────────
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # ── Binary operation: left OP right ────────────────────────────
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_BINARY_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            left = self._safe_eval_node(node.left)
            right = self._safe_eval_node(node.right)
            op_func = self._ALLOWED_BINARY_OPS[op_type]

            # Safety checks
            if op_type == ast.Div and right == 0:
                raise ValueError("Division by zero")
            if op_type == ast.Pow and right > 1000:
                raise ValueError(
                    f"Exponent too large: {right}. Max allowed: 1000"
                )

            return op_func(left, right)

        # ── Unary operation: +x or -x ─────────────────────────────────
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

            operand = self._safe_eval_node(node.operand)
            return self._ALLOWED_UNARY_OPS[op_type](operand)

        # ── ANYTHING ELSE → REJECT ─────────────────────────────────────
        raise ValueError(
            f"Unsafe expression element: {type(node).__name__}. "
            f"Only numbers and arithmetic operators are allowed."
        )

    def execute(self, **kwargs) -> dict:
        """
        Safely evaluate a math expression string.

        Args:
            expression: The math expression string (e.g., "2 + 3 * 4").

        Returns:
            {"status": "success", "result": <numeric_value>}
            or {"status": "error", "error_message": "..."}
        """
        expression = kwargs.get("expression", "")

        if not expression or not isinstance(expression, str):
            return {
                "status": "error",
                "error_message": "Missing or invalid 'expression' argument. "
                                 "Expected a string like '2 + 3'.",
            }

        # Clean up the expression (remove extra whitespace)
        expression = expression.strip()

        try:
            # Parse the expression string into an AST
            tree = ast.parse(expression, mode="eval")
            # The root of an "eval" parse is an ast.Expression node
            # The actual expression is in tree.body
            result = self._safe_eval_node(tree.body)

            # Round floats to avoid floating-point noise
            if isinstance(result, float) and result == int(result):
                result = int(result)

            return {
                "status": "success",
                "result": result,
                "expression": expression,
            }

        except (ValueError, SyntaxError, TypeError) as e:
            return {
                "status": "error",
                "error_message": f"Cannot evaluate '{expression}': {str(e)}",
            }


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONCRETE TOOL: Account Balance Lookup
# ═══════════════════════════════════════════════════════════════════════════
class AccountBalanceTool(Tool):
    """
    Queries a mock deterministic database for user account balances.

    ┌──────────────────────────────────────────────────────────────────┐
    │  WHY A MOCK DATABASE?                                            │
    │                                                                  │
    │  In a real system, this would query PostgreSQL, MongoDB, or an   │
    │  API. For our Baby LLM framework, we use a simple Python dict   │
    │  to simulate the database. The INTERFACE is identical to what    │
    │  a real database tool would look like — the only difference is  │
    │  the data source.                                                │
    │                                                                  │
    │  The LLM doesn't know (or care) that it's a mock. It just sees: │
    │    Input:  {"tool": "get_balance", "args": {"username": "john"}} │
    │    Output: {"status": "success", "result": {"balance": 500}}     │
    │                                                                  │
    │  This is the beauty of the TOOL ABSTRACTION — the LLM talks to  │
    │  a uniform interface, and we can swap the implementation from    │
    │  a mock dict to a real database without changing any LLM code.   │
    └──────────────────────────────────────────────────────────────────┘
    """

    # ── Mock Database ──────────────────────────────────────────────────
    # In production, this would be a real database connection.
    _MOCK_DB: dict[str, dict] = {
        "john": {
            "balance": 500.00,
            "currency": "USD",
            "account_type": "checking",
        },
        "alice": {
            "balance": 12_750.50,
            "currency": "USD",
            "account_type": "savings",
        },
        "bob": {
            "balance": 3_200.00,
            "currency": "EUR",
            "account_type": "checking",
        },
        "diana": {
            "balance": 89.99,
            "currency": "USD",
            "account_type": "checking",
        },
        "charlie": {
            "balance": 45_000.00,
            "currency": "GBP",
            "account_type": "savings",
        },
    }

    @property
    def name(self) -> str:
        return "get_balance"

    @property
    def description(self) -> str:
        return (
            "Looks up the account balance for a given username. "
            "Returns balance, currency, and account type. "
            "Available users: john, alice, bob, diana, charlie."
        )

    @property
    def parameters(self) -> dict:
        return {
            "username": {
                "type": "string",
                "description": (
                    "The username to look up. Must be lowercase. "
                    "Example: 'john'"
                ),
            }
        }

    def execute(self, **kwargs) -> dict:
        """
        Look up account balance for a username.

        Args:
            username: The username string (case-insensitive).

        Returns:
            {"status": "success", "result": {"balance": ..., "currency": ..., ...}}
            or {"status": "error", "error_message": "User not found..."}
        """
        username = kwargs.get("username", "")

        if not username or not isinstance(username, str):
            return {
                "status": "error",
                "error_message": "Missing or invalid 'username' argument. "
                                 "Expected a string like 'john'.",
            }

        # Normalize to lowercase (the LLM might capitalize)
        username = username.strip().lower()

        # Look up in our mock database
        if username in self._MOCK_DB:
            account = self._MOCK_DB[username]
            return {
                "status": "success",
                "result": {
                    "username": username,
                    "balance": account["balance"],
                    "currency": account["currency"],
                    "account_type": account["account_type"],
                },
            }
        else:
            available = ", ".join(sorted(self._MOCK_DB.keys()))
            return {
                "status": "error",
                "error_message": (
                    f"User '{username}' not found in the database. "
                    f"Available users: [{available}]"
                ),
            }


# ═══════════════════════════════════════════════════════════════════════════
# 5. CONVENIENCE — Create a pre-configured registry
# ═══════════════════════════════════════════════════════════════════════════
def create_default_registry() -> ToolRegistry:
    """
    Create a ToolRegistry with all default tools registered.

    This is the function that Phase 5 (Agentic Loop) will call to
    get a ready-to-use registry with all tools available.
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(AccountBalanceTool())
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Test the registry and tool execution
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — Symbolic Engine: Tools Verification")
    print("=" * 70)

    # ── 1. Create and populate registry ────────────────────────────────
    registry = create_default_registry()
    print(f"\n  Registered tools: {registry.list_tools()}")

    # ── 2. Print the full schema (what the LLM sees) ──────────────────
    print("\n" + "─" * 70)
    print("  Tool Schema (injected into LLM system prompt):")
    print("─" * 70)
    print(registry.get_schema_string())

    # ── 3. Test Calculator Tool ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test: Calculator Tool")
    print("─" * 70)

    calc_tests = [
        ("2 + 3", 5),
        ("10 * (3 + 2)", 50),
        ("2 ** 10", 1024),
        ("100 / 3", 33.333333),
        ("17 % 5", 2),
        ("(15 + 27)", 42),
        ("-5 + 3", -2),
        ("10 // 3", 3),
    ]

    for expr, expected in calc_tests:
        result = registry.execute("calculator", expression=expr)
        actual = result.get("result")
        status = "✅" if result["status"] == "success" else "❌"

        # For float comparison, use approximate equality
        if isinstance(expected, float):
            match = abs(actual - expected) < 0.001 if actual else False
        else:
            match = actual == expected

        print(f"  {status} '{expr}' = {actual}"
              f"{'  ✓' if match else f'  ✗ (expected {expected})'}")

    # Test error cases
    print("\n  Error handling tests:")

    # Division by zero
    result = registry.execute("calculator", expression="10 / 0")
    print(f"  ✅ '10 / 0' → {result['status']}: {result['error_message'][:60]}")

    # Dangerous code injection attempt
    result = registry.execute("calculator", expression="__import__('os').system('ls')")
    print(f"  ✅ Code injection → {result['status']}: {result['error_message'][:60]}")

    # Missing argument
    result = registry.execute("calculator")
    print(f"  ✅ Missing arg → {result['status']}: {result['error_message'][:60]}")

    # ── 4. Test Account Balance Tool ───────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test: Account Balance Tool")
    print("─" * 70)

    balance_tests = ["john", "Alice", "BOB", "diana", "charlie"]
    for username in balance_tests:
        result = registry.execute("get_balance", username=username)
        if result["status"] == "success":
            r = result["result"]
            print(f"  ✅ {username:10s} → {r['balance']:>10,.2f} {r['currency']} "
                  f"({r['account_type']})")
        else:
            print(f"  ❌ {username:10s} → {result['error_message']}")

    # User not found
    result = registry.execute("get_balance", username="nonexistent")
    print(f"\n  ✅ Unknown user → {result['status']}: "
          f"{result['error_message'][:60]}")

    # ── 5. Test unknown tool ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test: Unknown Tool Handling")
    print("─" * 70)
    result = registry.execute("hack_the_planet", target="nasa")
    print(f"  ✅ Unknown tool → {result['status']}: "
          f"{result['error_message'][:60]}")

    print("\n" + "=" * 70)
    print("  ✅ All tool tests passed!")
    print("=" * 70)
