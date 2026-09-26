"""Borrowed provider rows stay at root; explicit profile credentials stay private."""

import json
from pathlib import Path

import pytest

from agent.credential_pool import PooledCredential, load_pool


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    profiles = [root / "profiles" / name for name in ("coder", "reviewer")]
    for profile in profiles:
        profile.mkdir(parents=True)
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setattr(Path, "home", lambda: host_home)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    def use(home):
        monkeypatch.setenv("HERMES_HOME", str(home))

    use(root)
    return root, profiles, use


def _rows(home):
    path = home / "auth.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("credential_pool", {}).get("deepseek", [])


def _credential(credential_id):
    return PooledCredential(
        provider="deepseek", id=credential_id, label=credential_id,
        auth_type="api_key", priority=0, source="manual",
        access_token=f"sk-{credential_id}",
    )


def test_borrowed_pool_status_and_membership_follow_root(fleet, monkeypatch):
    root, (coder, reviewer), use = fleet
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-root-env")
    # A legacy unsanitized env row forces the load-time persistence path.
    root_row = {**_credential("shared").to_dict(),
                "source": "env:DEEPSEEK_API_KEY", "access_token": "sk-root-env"}
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"deepseek": [root_row]}}))

    use(coder)
    pool = load_pool("deepseek")
    assert pool.select().runtime_api_key == "sk-root-env"
    assert _rows(coder) == []
    assert not _rows(root)[0].get("access_token")
    assert pool.mark_exhausted_and_rotate(status_code=402, credential_id="shared") is None
    assert _rows(root)[0]["last_status"] == "exhausted"
    assert _rows(coder) == []

    use(reviewer)
    sibling = load_pool("deepseek")
    assert sibling.select() is None
    assert sibling.reset_statuses() == 1
    assert sibling.select().runtime_api_key == "sk-root-env"
    assert not _rows(root)[0]["last_status"]
    # Borrowers cannot delete the root's credential lifecycle.
    assert sibling.remove_index(1).id == "shared"
    assert [row["id"] for row in _rows(root)] == ["shared"]
    assert _rows(reviewer) == []

    use(root)
    load_pool("deepseek").add_entry(_credential("later"))
    use(coder)
    stale_pool = load_pool("deepseek")
    assert {entry.id for entry in stale_pool.entries()} == {"shared", "later"}

    # Root removes every row after this profile's snapshot. A stale status
    # update must not recreate either the root row or a profile-local copy.
    use(root)
    owner = load_pool("deepseek")
    assert owner.remove_index(2).id == "later"
    assert owner.remove_index(1).id == "shared"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    use(coder)
    stale_pool.mark_exhausted_and_rotate(status_code=402, credential_id="later")
    assert _rows(root) == []
    assert _rows(coder) == []
    use(reviewer)
    assert load_pool("deepseek").entries() == []


@pytest.mark.parametrize("root_present", [False, True])
@pytest.mark.parametrize("local_source", ["manual", "env"])
def test_profile_owned_credentials_preserve_isolation_and_removal(
    fleet, monkeypatch, root_present, local_source,
):
    root, (coder, reviewer), use = fleet
    if root_present:
        load_pool("deepseek").add_entry(_credential("shared"))
    root_rows = _rows(root)

    use(coder)
    if local_source == "env":
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-private")
    pool = load_pool("deepseek")
    if local_source == "manual":
        pool.add_entry(_credential("private"))
    local_rows = _rows(coder)
    assert [row["source"] for row in local_rows] == [
        "manual" if local_source == "manual" else "env:DEEPSEEK_API_KEY",
    ]
    local_id = local_rows[0]["id"]
    pool.mark_exhausted_and_rotate(status_code=402, credential_id=local_id)

    private = load_pool("deepseek")
    assert [entry.id for entry in private.entries()] == [local_id]
    assert private.select() is None
    assert private.reset_statuses() == 1
    assert private.select().runtime_api_key == "sk-private"
    assert private.remove_index(1).id == local_id
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert _rows(coder) == []
    assert _rows(root) == root_rows
    for home in (coder, reviewer):
        use(home)
        assert [entry.id for entry in load_pool("deepseek").entries()] == (
            ["shared"] if root_present else []
        )


def test_shell_copy_of_shared_key_does_not_shadow_or_bypass_cooldown(fleet, monkeypatch):
    root, (coder, reviewer), use = fleet
    load_pool("deepseek").add_entry(_credential("shared"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shared")
    use(coder)
    pool = load_pool("deepseek")
    assert [entry.id for entry in pool.entries()] == ["shared"]
    assert pool.mark_exhausted_and_rotate(status_code=402, credential_id="shared") is None
    assert _rows(coder) == []
    use(reviewer)
    assert load_pool("deepseek").select() is None
    assert _rows(reviewer) == []
    assert _rows(root)[0]["last_status"] == "exhausted"
