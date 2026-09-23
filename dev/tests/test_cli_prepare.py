from ftgspp.cli import main


def test_top_level_help_lists_prepare_subcommands(capsys):
    try:
        main(["--help"])
    except SystemExit as raised:
        assert raised.code == 0
    else:
        raise AssertionError("expected --help to exit")
    text = " ".join(capsys.readouterr().out.split())
    for name in ("viewer", "dataset", "masks", "points", "selfcap", "final-test"):
        assert name in text
    assert "ftgspp prepare --help" in text


def test_prepare_help_lists_subcommands_not_selfcap_only(capsys):
    main(["prepare", "--help"])
    text = capsys.readouterr().out
    for name in ("viewer", "dataset", "masks", "points", "selfcap", "final-test"):
        assert name in text
    assert "SelfCap layout adapter" in text
    assert "--scene-root" not in text


def test_prepare_without_subcommand_prints_help(capsys):
    main(["prepare"])
    text = capsys.readouterr().out
    assert "usage: ftgspp prepare" in text
    assert "dataset" in text
    assert "viewer" in text
    assert "selfcap" in text


def test_unknown_prepare_command_exits(capsys):
    try:
        main(["prepare", "unknown-tool"])
    except SystemExit as raised:
        assert raised.code == 2
    else:
        raise AssertionError("expected unknown prepare command to exit")
    err = capsys.readouterr().err
    assert "unknown prepare command" in err
    assert "dataset" in err


def test_flag_only_prepare_warns_and_forwards_to_selfcap(monkeypatch, capsys):
    captured: dict[str, list[str]] = {}

    def fake_selfcap(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr("ftgspp.data.prepare_selfcap.main", fake_selfcap)
    main(["prepare", "--scene-root", "data/scene"])
    assert captured["argv"] == ["--scene-root", "data/scene"]
    assert "prepare selfcap" in capsys.readouterr().err
