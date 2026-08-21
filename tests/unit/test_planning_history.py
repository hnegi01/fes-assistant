"""
Unit tests for _build_planning_history().

Covers: turn capping, assistant content-only stripping, empty message skipping,
latest_user_message exclusion, and zero-turn (disabled) mode.
"""

import backend.agent.llm_agent as m


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text, tool_result=None):
    msg = {"role": "assistant", "content": text}
    if tool_result:
        msg["tool_result"] = tool_result
    return msg


class TestBuildPlanningHistory:
    def test_excludes_latest_user_message(self):
        latest = _user("latest question")
        messages = [_user("first"), _assistant("first reply"), latest]

        result = m._build_planning_history(messages, latest, n_turns=5)

        contents = [m["content"] for m in result]
        assert "latest question" not in contents

    def test_includes_prior_user_and_assistant(self):
        latest = _user("now")
        messages = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2"), latest]

        result = m._build_planning_history(messages, latest, n_turns=5)

        assert len(result) == 4
        assert result[0] == {"role": "user", "content": "q1"}
        assert result[1] == {"role": "assistant", "content": "a1"}

    def test_strips_tool_result_from_assistant(self):
        latest = _user("now")
        messages = [
            _user("show dashboards"),
            _assistant("Found 5 dashboards.", tool_result={"ok": True, "result": [{"id": 1}]}),
            latest,
        ]

        result = m._build_planning_history(messages, latest, n_turns=5)

        asst = next(m for m in result if m["role"] == "assistant")
        assert "tool_result" not in asst
        assert asst["content"] == "Found 5 dashboards."

    def test_caps_at_n_turns(self):
        latest = _user("turn6")
        messages = []
        for i in range(1, 6):
            messages += [_user(f"q{i}"), _assistant(f"a{i}")]
        messages.append(latest)

        result = m._build_planning_history(messages, latest, n_turns=3)

        # 3 turns * 2 messages = 6
        assert len(result) == 6
        assert result[0]["content"] == "q3"

    def test_skips_empty_assistant_messages(self):
        latest = _user("now")
        messages = [
            _user("q1"),
            _assistant(""),  # empty — pending confirmation turn
            _assistant("a1"),
            latest,
        ]

        result = m._build_planning_history(messages, latest, n_turns=5)

        contents = [m["content"] for m in result]
        assert "" not in contents
        assert "a1" in contents

    def test_zero_turns_returns_empty(self):
        latest = _user("now")
        messages = [_user("q1"), _assistant("a1"), latest]

        result = m._build_planning_history(messages, latest, n_turns=0)

        assert result == []

    def test_no_prior_messages_returns_empty(self):
        latest = _user("first ever message")
        result = m._build_planning_history([latest], latest, n_turns=5)
        assert result == []

    def test_ignores_non_user_assistant_roles(self):
        latest = _user("now")
        messages = [
            {"role": "system", "content": "system prompt"},
            _user("q1"),
            _assistant("a1"),
            latest,
        ]

        result = m._build_planning_history(messages, latest, n_turns=5)

        roles = [m["role"] for m in result]
        assert "system" not in roles
