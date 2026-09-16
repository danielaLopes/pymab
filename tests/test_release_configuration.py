from pathlib import Path

from pytest import MonkeyPatch

from scripts import check_versions


def test_release_configuration_is_synchronized() -> None:
    assert check_versions.release_configuration_errors() == []


def test_release_configuration_rejects_hard_coded_browser_version(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    browser_test = tmp_path / "runtime-smoke.spec.ts"
    browser_test.write_text(
        'fetch("runtime-manifest.json");\n'
        'const pymabVersion = "2.0.0";\n'
        'page.getByText("2.0.0", { exact: true });\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check_versions, "WEB_RUNTIME_SMOKE", browser_test)

    errors = check_versions.release_configuration_errors()

    assert any(
        "must not contain a hard-coded release version" in error for error in errors
    )


def test_release_configuration_rejects_unvalidated_release_checkout(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    workflow = tmp_path / "release.yml"
    current = (check_versions.ROOT / check_versions.PUBLISH_WORKFLOW).read_text(
        encoding="utf-8"
    )
    workflow.write_text(
        current.replace(
            "ref: ${{ needs.metadata.outputs.tag }}",
            "ref: ${{ github.event.release.tag_name }}",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_versions, "PUBLISH_WORKFLOW", workflow)

    errors = check_versions.release_configuration_errors()

    assert any("must use the validated metadata tag" in error for error in errors)
