import csv, json

from src.backtest.top_down_swing_v2_portfolio import _rank, _read_signals


def candidate(sid, location, rr, target="DAILY", strength="MEANINGFUL", confirmation="SUPPORT_REACTION"):
    return {"security_id":sid,"setup_instance_id":sid,"daily_location_state":location,
        "plan":{"daily_support_strength":"MEANINGFUL","planned_reward_r":rr,
                "target_timeframe":target,"target_zone_strength":strength,"volume_confirmation_passed":True},
        "result":{"confirmation_type":confirmation}}


def test_v2_allocator_priority_is_ex_ante_and_deterministic():
    near=candidate("A","NEAR_RECENT_DAILY_SUPPORT",3.0)
    inside=candidate("Z","INSIDE_RECENT_DAILY_SUPPORT",1.8)
    better_target=candidate("B","INSIDE_RECENT_DAILY_SUPPORT",2.0,"WEEKLY","MAJOR")
    assert sorted([near,better_target,inside],key=_rank)==[better_target,inside,near]
    assert sorted([near,better_target,inside],key=_rank)==sorted([near,better_target,inside],key=_rank)


def test_signal_reader_decodes_top_level_support_for_portfolio_diagnostics(tmp_path):
    path=tmp_path/"signals.csv"
    fields=["result","plan","support","resistance","setup_instance_id","setup_signal_date","symbol"]
    row={"result":json.dumps({"plan":{"entry":11},"entry_timestamp":"2024-01-02T11:15:00+05:30",
                              "confirmation_timestamp":"2024-01-02T10:15:00+05:30"}),
         "plan":json.dumps({"entry":11}),"support":json.dumps({"lower_bound":9,"upper_bound":10}),
         "resistance":"","setup_instance_id":"S1","setup_signal_date":"2024-01-02","symbol":"TEST"}
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow(row)
    signal=_read_signals(path)[0]
    assert signal["support"]=={"lower_bound":9,"upper_bound":10}
