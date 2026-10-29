import csv
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest

import pydysp
from pydysp.campaign import Campaign
from pydysp.channel import Channel
from pydysp.test import Test

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def make_test(
    name: str = "T",
    channels: tuple[str, ...] = ("Input", "Response"),
    n: int = 200,
    dt: float = 0.01,
    tags: set[str] | None = None,
    meta: dict | None = None,
    description: str | None = None,
    source_file: str | None = None,
) -> Test:
    """Build a small synthetic Test with sinusoidal channels."""
    t = np.arange(n) * dt
    chans = [
        Channel(data=np.sin(2 * np.pi * (i + 1) * t), dt=dt, name_user=c)
        for i, c in enumerate(channels)
    ]
    return Test(
        name=name,
        description=description,
        source_file=source_file,
        channels=chans,
        tags=tags or set(),
        meta=meta or {},
    )


def fake_loader(path: str) -> Test:
    """Loader that builds a synthetic Test named after the file stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return make_test(
        name=stem,
        description=f"loaded {stem}",
        source_file=path,
        tags={"loaded"},
        meta={"loader_key": 1, "header": {"bulky": [1, 2, 3]}},
    )


def touch(folder, *names):
    """Create empty files and return their paths."""
    paths = []
    for nm in names:
        p = os.path.join(folder, nm)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("")
        paths.append(p)
    return paths


def simple_campaign(numbers=None) -> Campaign:
    tests = [
        make_test("Test 001", tags={"hammer"}, description="Hammer"),
        make_test("Test 002", tags={"white-noise"}, meta={"amp": 0.06}),
        make_test("Test 003", tags={"record"}, meta={"amp": 0.2}),
        make_test("Test 004", tags={"white-noise", "record"}, meta={"amp": 0.1}),
    ]
    return Campaign.from_tests(tests, numbers=numbers, name="Camp")


# ----------------------------------------------------------------------
# Construction and numbering
# ----------------------------------------------------------------------


def test_exported_from_package():
    assert pydysp.Campaign is Campaign
    assert "Campaign" in pydysp.__all__


def test_from_tests_default_numbers_and_container_protocol():
    camp = simple_campaign()
    assert camp.numbers == [1, 2, 3, 4]
    assert len(camp) == 4
    assert [t.name for t in camp] == camp.names
    assert camp[0].name == "Test 001"
    assert camp[-1].name == "Test 004"
    assert camp.name == "Camp"
    assert "n_tests=4" in repr(camp)


def test_from_tests_explicit_numbers_and_errors():
    camp = simple_campaign(numbers=[29, 30, 31, 33])
    assert camp.numbers == [29, 30, 31, 33]
    assert camp.get(number=31).name == "Test 003"
    with pytest.raises(ValueError, match="numbers has 2 entries"):
        Campaign.from_tests([make_test(), make_test()][:1] * 3, numbers=[1, 2])
    with pytest.raises(ValueError, match="unique"):
        Campaign.from_tests([make_test(), make_test()], numbers=[5, 5])
    with pytest.raises(TypeError):
        Campaign(tests=["not a test"])


def test_getitem_slice_returns_campaign_with_numbers():
    camp = simple_campaign(numbers=[10, 11, 12, 13])
    sub = camp[1:3]
    assert isinstance(sub, Campaign)
    assert sub.numbers == [11, 12]
    with pytest.raises(TypeError):
        camp["Test 001"]


def test_get_by_name_and_errors():
    camp = simple_campaign()
    assert camp.get(name="test 002").meta["amp"] == 0.06
    with pytest.raises(KeyError):
        camp.get(number=99)
    with pytest.raises(KeyError):
        camp.get(name="nope")
    with pytest.raises(ValueError):
        camp.get()
    with pytest.raises(ValueError):
        camp.get(number=1, name="Test 001")


def test_from_files_metadata_and_merging(tmp_path):
    files = touch(tmp_path, "a.mat", "b.mat", "c.mat")
    camp = Campaign.from_files(
        files,
        loader=fake_loader,
        descriptions=["Hammer", None, "Record 901"],
        tags=[{"hammer"}, "white-noise", None],
        meta={"amp": [None, 0.06, 0.2], "record": ["-", "-", "901"]},
        name="C",
    )
    assert camp.numbers == [1, 2, 3]
    # Names kept from loader when not given
    assert camp.names == ["a", "b", "c"]
    # Descriptions replaced only where given
    assert camp.descriptions == ["Hammer", "loaded b", "Record 901"]
    # Tags are added to the loader's tags
    assert camp.tags == [{"loaded", "hammer"}, {"loaded", "white-noise"}, {"loaded"}]
    # Meta merged; None values skipped
    assert "amp" not in camp[0].meta
    assert camp.meta_values("amp") == [None, 0.06, 0.2]
    assert camp.meta_values("amp", default=0.0)[0] == 0.0
    assert camp.meta_values("loader_key") == [1, 1, 1]
    assert camp[2].meta["record"] == "901"


def test_from_files_meta_as_rows(tmp_path):
    files = touch(tmp_path, "a.mat", "b.mat")
    camp = Campaign.from_files(files, loader=fake_loader, meta=[{"amp": 0.06}, None])
    assert camp.meta_values("amp") == [0.06, None]


@pytest.mark.parametrize(
    "kwargs, label",
    [
        ({"names": ["x"]}, "names"),
        ({"descriptions": ["x", "y", "z"]}, "descriptions"),
        ({"tags": [{"x"}]}, "tags"),
        ({"numbers": [1]}, "numbers"),
        ({"meta": {"amp": [1, 2, 3]}}, r"meta\['amp'\]"),
        ({"meta": [{"amp": 1}]}, "meta"),
    ],
)
def test_from_files_length_mismatch_names_list(tmp_path, kwargs, label):
    files = touch(tmp_path, "a.mat", "b.mat")
    with pytest.raises(ValueError, match=f"^{label} has"):
        Campaign.from_files(files, loader=fake_loader, **kwargs)


def test_from_files_conflicting_arguments(tmp_path):
    files = touch(tmp_path, "a.mat")
    with pytest.raises(ValueError, match="names= or name_template="):
        Campaign.from_files(
            files, loader=fake_loader, names=["x"], name_template="{stem}"
        )
    with pytest.raises(ValueError, match="numbers= or number_from="):
        Campaign.from_files(
            files, loader=fake_loader, numbers=[1], number_from="{number}"
        )


def test_from_files_loader_must_return_test(tmp_path):
    files = touch(tmp_path, "a.mat")
    with pytest.raises(TypeError, match="expected a Test"):
        Campaign.from_files(files, loader=lambda p: "nope")


def test_number_from_and_name_template(tmp_path):
    files = touch(
        tmp_path,
        "SH2601_29_WN.mat",
        "SH2601_30_record.MAT",
        "SH2601_32_record.mat",  # gap: test 31 was not recorded
    )
    camp = Campaign.from_files(
        files,
        loader=fake_loader,
        number_from="sh2601_{number}_*",
        name_template="Shallow Foundation - Test {number:03d} ({stem})",
    )
    assert camp.numbers == [29, 30, 32]
    assert camp.names[0] == "Shallow Foundation - Test 029 (SH2601_29_WN)"


def test_number_from_matches_stem_without_extension(tmp_path):
    files = touch(tmp_path, "Shear_001_1000Hz.MAT", "Shear_002_1000Hz.MAT")
    camp = Campaign.from_files(
        files, loader=fake_loader, number_from="Shear_{number}_1000Hz"
    )
    assert camp.numbers == [1, 2]


def test_number_from_errors(tmp_path):
    files = touch(tmp_path, "Shear_001_1000Hz.mat", "other.mat")
    with pytest.raises(ValueError, match="'other' does not match"):
        Campaign.from_files(files, loader=fake_loader, number_from="Shear_{number}*")
    with pytest.raises(ValueError, match="exactly once"):
        Campaign.from_files(files, loader=fake_loader, number_from="Shear_*")
    dup = touch(tmp_path, "T_1_a.mat", "T_01_b.mat")
    with pytest.raises(ValueError, match="unique"):
        Campaign.from_files(dup, loader=fake_loader, number_from="T_{number}_*")


def test_name_template_bad_field(tmp_path):
    files = touch(tmp_path, "a.mat")
    with pytest.raises(ValueError, match="name_template"):
        Campaign.from_files(files, loader=fake_loader, name_template="{idx}")


def test_from_folder_natural_sort_and_case_insensitive(tmp_path):
    touch(tmp_path, "Test_10.mat", "Test_2.MAT", "Test_1.Mat", "notes.txt")
    camp = Campaign.from_folder(tmp_path, loader=fake_loader)
    assert camp.names == ["Test_1", "Test_2", "Test_10"]
    assert camp.numbers == [1, 2, 3]
    assert camp.name == os.path.basename(tmp_path)


def test_from_folder_lists_follow_sorted_order(tmp_path):
    touch(tmp_path, "Test_10.mat", "Test_2.mat")
    camp = Campaign.from_folder(
        tmp_path,
        loader=fake_loader,
        number_from="Test_{number}",
        descriptions=["second", "tenth"],
        name="My campaign",
    )
    assert camp.numbers == [2, 10]
    assert camp.descriptions == ["second", "tenth"]
    assert camp.name == "My campaign"


def test_from_folder_recursive_and_errors(tmp_path):
    touch(tmp_path, "day2/T_1.mat", "day1/T_10.mat", "day1/T_2.mat", "x_0.mat")
    flat = Campaign.from_folder(tmp_path, loader=fake_loader)
    assert flat.names == ["x_0"]
    # Sorted by path relative to the folder, in natural order
    deep = Campaign.from_folder(tmp_path, loader=fake_loader, recursive=True)
    assert deep.names == ["T_2", "T_10", "T_1", "x_0"]
    with pytest.raises(ValueError, match="No files matching"):
        Campaign.from_folder(tmp_path, pattern="*.csv", loader=fake_loader)
    with pytest.raises(FileNotFoundError):
        Campaign.from_folder(tmp_path / "missing", loader=fake_loader)


def test_from_folder_with_real_loader(tmp_path):
    # Round trip through Test.to_csv / Test.from_csv to use a real loader
    for i in (1, 2):
        make_test(f"T{i}").to_csv(str(tmp_path / f"run_{i}.csv"))
    camp = Campaign.from_folder(
        tmp_path, pattern="*.csv", loader=Test.from_csv, number_from="run_{number}"
    )
    assert camp.numbers == [1, 2]
    assert camp[0].n_channels == 2


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def test_select_keeps_numbers_and_returns_new_campaign():
    camp = simple_campaign(numbers=[29, 30, 31, 33])
    sub = camp.select(indices=slice(2, None))
    assert sub.numbers == [31, 33]
    assert sub[0].name == "Test 003"  # index recomputed
    assert camp.numbers == [29, 30, 31, 33]  # original unchanged
    assert sub.select(numbers=33).numbers == [33]


def test_select_by_indices_list_and_errors():
    camp = simple_campaign()
    assert camp.select(indices=[3, 0]).numbers == [1, 4]  # campaign order
    assert camp.select(indices=-1).numbers == [4]
    with pytest.raises(IndexError):
        camp.select(indices=[7])


def test_select_numbers_with_gaps_ignored():
    camp = simple_campaign(numbers=[29, 30, 31, 33])
    assert camp.select(numbers=range(29, 40)).numbers == [29, 30, 31, 33]
    assert camp.select(numbers=range(31, 33)).numbers == [31]


def test_select_names_raises_on_unknown():
    camp = simple_campaign()
    assert camp.select(names=["test 003"]).names == ["Test 003"]
    assert camp.select(names="Test 001").numbers == [1]
    with pytest.raises(KeyError, match="Test 099"):
        camp.select(names=["Test 001", "Test 099"])


def test_select_tags_where_and_combined():
    camp = simple_campaign()
    assert camp.select(tags={"white-noise"}).numbers == [2, 4]
    assert camp.select(tags={"hammer", "record"}).numbers == [1, 3, 4]
    assert camp.select(exclude_tags={"hammer"}).numbers == [2, 3, 4]
    assert camp.select(where=lambda t: t.meta.get("amp", 0) > 0.08).numbers == [3, 4]
    combined = camp.select(tags="record", exclude_tags="white-noise", numbers=[3, 4])
    assert combined.numbers == [3]


def test_select_empty_warns():
    camp = simple_campaign()
    with pytest.warns(UserWarning, match="empty"):
        sub = camp.select(tags={"nothing"})
    assert len(sub) == 0
    with pytest.warns(UserWarning):
        camp.select(numbers=[100])


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def test_info_returns_table_string():
    camp = simple_campaign(numbers=[5, 6, 7, 8])
    s = camp.info()
    assert isinstance(s, str)
    assert "Campaign: Camp" in s
    assert "Test 003" in s and "white-noise" in s
    for col in ("number", "description", "tags", "duration", "channels"):
        assert col in s


# ----------------------------------------------------------------------
# Test-info CSV
# ----------------------------------------------------------------------


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_test_info_csv_export_columns(tmp_path):
    camp = simple_campaign(numbers=[1, 3, 4, 6])
    camp = Campaign.from_tests(
        [
            *camp.tests[:3],
            make_test(
                "Test 006",
                meta={
                    "amp": 0.1,
                    "note": "x" * 200,  # too long: skipped
                    "header": {"a": 1},  # bulky: skipped
                    "same": "s",
                },
                tags={"a", "b"},
            ),
        ],
        numbers=[1, 3, 4, 6],
    )
    out = tmp_path / "tests.csv"
    camp.test_info_to_csv(str(out))
    rows = read_csv(out)
    header = rows[0]
    # number and name always; empty timestamp/source_file omitted
    assert header == ["number", "name", "description", "tags", "amp", "same"]
    assert rows[1][:4] == ["1", "Test 001", "Hammer", "hammer"]
    assert rows[1][4] == ""  # no amp for test 1
    assert rows[4][3] == "a,b"
    with pytest.raises(FileExistsError):
        camp.test_info_to_csv(str(out), overwrite=False)


def test_test_info_csv_keeps_identical_columns(tmp_path):
    tests = [make_test(f"T{i}", description="Same", meta={"k": 1}) for i in (1, 2)]
    out = tmp_path / "tests.csv"
    Campaign.from_tests(tests).test_info_to_csv(str(out))
    assert read_csv(out)[0] == ["number", "name", "description", "k"]


def test_test_info_csv_round_trip(tmp_path):
    files = touch(tmp_path, "SH_29.mat", "SH_30.mat", "SH_32.mat")
    camp = Campaign.from_files(
        files,
        loader=fake_loader,
        number_from="SH_{number}",
        descriptions=["Hammer", "WN 0.06g", "Record 901"],
        tags=[{"hammer"}, {"white-noise"}, {"record", "main"}],
        meta={"amp": [None, 0.06, 0.2], "record": [None, None, "901"]},
    )
    out = tmp_path / "tests.csv"
    camp.test_info_to_csv(str(out))
    header = read_csv(out)[0]
    assert "source_file" in header and "loader_key" in header
    assert "header" not in header

    fresh = Campaign.from_files(files, loader=fake_loader, number_from="SH_{number}")
    back = fresh.with_test_info_from_csv(str(out))
    assert back.numbers == camp.numbers
    assert back.names == camp.names
    assert back.descriptions == camp.descriptions
    assert back.tags == camp.tags
    assert back.meta_values("amp") == [None, 0.06, 0.2]
    # "901" is numeric text: the whole column parses as int
    assert back.meta_values("record") == [None, None, 901]
    assert back.meta_values("loader_key") == [1, 1, 1]
    # Bulky meta from the loader is untouched
    assert back[0].meta["header"] == {"bulky": [1, 2, 3]}
    # Original is not modified
    assert fresh.descriptions[0] == "loaded SH_29"


def test_with_test_info_from_csv_flexible_headers_and_types(tmp_path):
    camp = simple_campaign(numbers=[1, 3, 4, 6])
    out = tmp_path / "info.csv"
    out.write_text(
        "Test Number,Description,TAG,Amp,Comment,Count\n"
        "3,White Noise 0.06g,white-noise; wn,0.06,first,2\n"
        "4,,,1e-1,,\n"
        "6,Record,record,0.2,third,5\n"
    )
    new = camp.with_test_info_from_csv(str(out))
    assert new.get(number=3).description == "White Noise 0.06g"
    assert new.get(number=3).tags == {"white-noise", "wn"}
    # Empty cells leave values unchanged
    assert new.get(number=4).description == camp.get(number=4).description
    assert new.get(number=4).tags == camp.get(number=4).tags
    # Untouched test
    assert new.get(number=1) is camp.get(number=1)
    assert new.meta_values("Amp") == [None, 0.06, 0.1, 0.2]
    assert new.meta_values("Comment") == [None, "first", None, "third"]
    assert new.meta_values("Count") == [None, 2, None, 5]
    assert isinstance(new.get(number=6).meta["Count"], int)


def test_with_test_info_from_csv_match_by_name(tmp_path):
    camp = simple_campaign()
    out = tmp_path / "info.csv"
    out.write_text("name,description\ntest 002,Renamed description\n")
    new = camp.with_test_info_from_csv(str(out))
    assert new.descriptions[1] == "Renamed description"


def test_with_test_info_from_csv_number_renames(tmp_path):
    camp = simple_campaign()
    out = tmp_path / "info.csv"
    out.write_text("number,name\n2,WN 1\n")
    assert camp.with_test_info_from_csv(str(out)).names[1] == "WN 1"


@pytest.mark.parametrize(
    "content, match",
    [
        ("number,description\n9,x\n", "could not match"),
        ("name,description\nnobody,x\n", "could not match"),
        ("number,description\nabc,x\n", "valid test number"),
        ("number,description\n1,x\n1,y\n", "more than one row"),
        ("description,amp\nx,1\n", "'number' or a 'name'"),
        ("", "empty"),
    ],
)
def test_with_test_info_from_csv_errors(tmp_path, content, match):
    out = tmp_path / "info.csv"
    out.write_text(content)
    with pytest.raises(ValueError, match=match):
        simple_campaign().with_test_info_from_csv(str(out))


def test_with_test_info_from_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        simple_campaign().with_test_info_from_csv(str(tmp_path / "none.csv"))


# ----------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------


def sensor_change_campaign() -> Campaign:
    """Tests 1-3 record A12 on DAQ ch3; tests 4-6 record its replacement on ch4."""
    tests = []
    for i in range(1, 7):
        t = make_test(f"T{i}", channels=("ch1", "ch2", "ch3", "ch4"))
        tests.append(t)
    return Campaign.from_tests(tests, numbers=[1, 2, 3, 4, 5, 6])


def write_channel_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name_input", "name_user"])
        w.writerows(rows)


def test_with_channel_info_from_csv_by_number_range(tmp_path):
    camp = Campaign.from_tests(
        [
            Test(
                name=f"T{i}",
                channels=[
                    Channel(data=np.zeros(10), dt=0.1, name_input=f"CH{k}")
                    for k in range(1, 5)
                ],
            )
            for i in range(1, 7)
        ]
    )
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_channel_csv(a, [["CH1", "Table"], ["CH3", "A12"], ["CH4", "A12_dead"]])
    write_channel_csv(b, [["CH1", "Table"], ["CH4", "A12"], ["CH3", "A12_dead"]])
    camp = camp.with_channel_info_from_csv(str(a), numbers=range(1, 4))
    camp = camp.with_channel_info_from_csv(str(b), numbers=range(4, 7))
    for t in camp:
        assert t["A12"].name_input == ("CH3" if t.name in {"T1", "T2", "T3"} else "CH4")
    assert camp.missing_channels() == {}
    assert "All 4 channels are present in all 6 tests." in camp.check_channels()


def test_with_channel_info_from_csv_defaults_to_all_and_warns(tmp_path):
    camp = sensor_change_campaign()
    a = tmp_path / "a.csv"
    write_channel_csv(a, [["ch1", "Table"]])
    new = camp.with_channel_info_from_csv(str(a))
    assert all(t[0].name_user == "Table" for t in new)
    assert camp[0][0].name_user == "ch1"  # original unchanged
    with pytest.warns(UserWarning, match="No test"):
        camp.with_channel_info_from_csv(str(a), numbers=[99])


def test_missing_channels_and_report():
    def t(name, chans):
        return make_test(name, channels=chans)

    base = ("A1", "A12")
    tests = [t(f"T{i}", base) for i in range(1, 14)]
    tests += [t("T14", ("A1", "A4b")), t("T15", ("A1", "A4b"))]
    tests += [t(f"T{i}", ("A1", "A12")) for i in range(16, 30)]
    tests += [t(f"T{i}", ("A1", "A12", "A4b")) for i in range(30, 36)]
    numbers = list(range(1, 36))
    camp = Campaign.from_tests(tests, numbers=numbers, name="C")

    missing = camp.missing_channels()
    assert missing == {
        "A12": [14, 15],
        "A4b": list(range(1, 14)) + list(range(16, 30)),
    }
    report = camp.check_channels()
    # The shorter of "missing in" and "present only in" is reported
    assert "Channel A12: missing in tests 14, 15" in report
    assert "Channel A4b: present only in tests 14, 15, 30-35" in report


def test_check_channels_labels_by_real_numbers_and_runs():
    tests = [
        make_test("a", channels=("X",)),
        make_test("b", channels=("X", "Y")),
        make_test("c", channels=("X", "Y")),
        make_test("d", channels=("X",)),
        make_test("e", channels=("X", "Y")),
    ]
    # Numbers have a gap (31 not recorded); runs follow campaign order
    camp = Campaign.from_tests(tests, numbers=[29, 30, 32, 33, 34])
    assert camp.missing_channels() == {"Y": [29, 33]}
    assert "Channel Y: missing in tests 29, 33" in camp.check_channels()
    tests[0] = make_test("a", channels=("X", "Y"))
    tests[3] = make_test("d", channels=("X", "Y"))
    tests.append(make_test("f", channels=("X",)))
    tests.append(make_test("g", channels=("X",)))
    tests.append(make_test("h", channels=("X",)))
    camp = Campaign.from_tests(tests, numbers=[29, 30, 32, 33, 34, 40, 41, 42])
    assert "Channel Y: missing in tests 40-42" in camp.check_channels()


def test_check_channels_reports_duplicates_case_insensitive():
    camp = Campaign.from_tests(
        [
            make_test("a", channels=("A1", "a1", "B")),
            make_test("b", channels=("A1", "B")),
        ],
        numbers=[7, 8],
    )
    assert camp.missing_channels() == {}
    report = camp.check_channels()
    assert "Test 7: duplicate channel name(s) A1" in report
    assert "All" not in report


def test_no_warnings_for_normal_use(tmp_path):
    camp = simple_campaign()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        camp.select(numbers=[1, 2])
        camp.info()
        camp.check_channels()


# ----------------------------------------------------------------------
# info(): durations without processing channel data
# ----------------------------------------------------------------------


def info_durations(camp: Campaign) -> list[str]:
    """Duration column of camp.info(), one string per test."""
    lines = [ln for ln in camp.info().splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    col = header.index("duration [s]")
    return [ln.strip("|").split("|")[col].strip() for ln in lines[2:]]


def processing_test(name: str = "P") -> Test:
    """Test whose channels have drift correction and filtering set."""
    return (
        make_test(name, channels=("A1", "A2", "A3")).drift_corrected().filtered(fc=10.0)
    )


def test_info_does_not_process_channel_data(monkeypatch):
    camp = Campaign.from_tests(
        [processing_test("a"), processing_test("b").trimmed(t_start=0.5, t_end=1.5)]
    )
    calls = []
    original = Channel.processed

    def spy(self, *args, **kwargs):
        calls.append(self.name_user)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Channel, "processed", spy)
    camp.info()
    assert calls == []
    # Test.duration, by contrast, processes every channel
    assert camp[0].duration > 0
    assert sorted(calls) == ["A1", "A2", "A3"]


def test_info_duration_unchanged_for_normal_and_trimmed_tests():
    normal = processing_test("normal")
    trimmed = processing_test("trimmed").trimmed(t_start=0.5, t_end=1.25)
    camp = Campaign.from_tests([normal, trimmed])
    assert info_durations(camp) == [f"{normal.duration:.6g}", f"{trimmed.duration:.6g}"]
    assert info_durations(camp) == ["1.99", "0.75"]
    assert "nan" not in camp.info()


def test_info_mixed_trims_give_nan_and_note():
    t = processing_test("mixed")
    mixed = t.trimmed(selector="A2", t_start=0.5, t_end=1.0)
    with pytest.raises(ValueError):
        _ = mixed.duration  # Test.duration refuses
    camp = Campaign.from_tests([t, mixed], numbers=[3, 7])
    info = camp.info()  # does not raise
    assert info_durations(camp) == ["1.99", "nan"]
    assert "nan: the channels of test(s) 7 do not all have the same duration." in info


def test_info_invalid_trim_and_empty_test():
    bad = make_test("bad", channels=("A1",))
    bad = Test(
        name="bad",
        channels=[replace(bad[0], trim_params={"t_start": 50.0, "t_end": 60.0})],
    )
    empty = Test(name="empty")
    camp = Campaign.from_tests([bad, empty])
    assert info_durations(camp) == ["-", "0"]
