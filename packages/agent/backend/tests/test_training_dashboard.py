from scripts.training_dashboard import format_dashboard


def test_format_dashboard_renders_progress_bar():
    stats = {
        "sft_count": 150,
        "dpo_count": 20,
        "threads_processed": 30,
        "threads_with_feedback": 18,
    }
    output = format_dashboard(stats, target_sft=800)
    assert "150" in output
    assert "800" in output
    # 150/800 = 18.75% → accept 18 or 19
    assert any(p in output for p in ("18%", "18.75%", "19%"))


def test_format_dashboard_handles_zero():
    stats = {
        "sft_count": 0,
        "dpo_count": 0,
        "threads_processed": 0,
        "threads_with_feedback": 0,
    }
    output = format_dashboard(stats, target_sft=800)
    assert "0" in output


def test_format_dashboard_shows_remaining():
    stats = {
        "sft_count": 300,
        "dpo_count": 50,
        "threads_processed": 60,
        "threads_with_feedback": 40,
    }
    output = format_dashboard(stats, target_sft=800)
    assert "500" in output  # 800 - 300 = 500 remaining
