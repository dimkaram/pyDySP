# test.py
from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

# typing.Sequence kept: it is part of the public ChannelSelector alias below.
from typing import (  # noqa: UP035
    Any,
    Literal,
    Sequence,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tabulate import tabulate

from .channel import Channel

# Type aliases for channel selection / referencing
# Public aliases kept as typing.Union: "|" would change their runtime type.
ChannelKey = Union[int, str, Channel]  # noqa: UP007
ChannelSelector = Union[ChannelKey, Sequence[ChannelKey], slice, None]  # noqa: UP007


@dataclass
class Test:
    """
    Container for an experiment composed of multiple time-history channels.

    The ``Test`` object stores an ordered list of :class:`Channel` instances
    together with test-level metadata (name, description, source file,
    timestamp, tags, and arbitrary ``meta``). It provides helpers for
    selecting channels, batch processing (drift/filter/baseline/trim),
    pairwise spectral analyses, simple modal identification glue, plotting,
    and common I/O routines (MAT/CSV).

    Parameters
    ----------
    name : str
        Human-readable name for the test.
    description : str, optional
        Longer description of the test.
    source_file : str, optional
        Path or identifier of the primary data file used to build this test.
    timestamp : str, optional
        String representation of the test date/time.
    channels : list of Channel, optional
        Ordered list of :class:`Channel` objects belonging to this test.
    tags : set of str, optional
        Free-form tags for grouping and filtering tests.
    meta : dict, optional
        Free-form metadata dictionary.
    """

    __test__ = False  # to avoid pytest collecting this as a test case

    # Name of the experiment (e.g. 'Shaking table Test 07').
    name: str
    # Longer description.
    description: str | None = None
    # Path or identifier of the primary raw data file from which this Test was built.
    source_file: str | None = None
    # String representation of the test time/date
    timestamp: str | None = None

    # Ordered list of Channel objects belonging to this experiment.
    channels: list[Channel] = field(default_factory=list)
    # Free-form tags for grouping and filtering tests.
    tags: set[str] = field(default_factory=set)
    # Free-form metadata dictionary.
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Normalise internal containers and enforce basic type assumptions.

        Notes
        -----
        This method is called automatically after dataclass initialisation.
        It ensures that:

        * ``channels`` is a list of :class:`Channel` objects.
        * ``tags`` is stored as a set.
        * ``meta`` is stored as a plain dictionary.
        * If ``name`` is empty and ``source_file`` is provided, a name is
          derived from the file stem.
        """
        # Always store channels as a list of Channel objects
        self.channels = list(self.channels)
        # Confirm all entries are Channel instances
        for i, ch in enumerate(self.channels):
            if not isinstance(ch, Channel):
                raise TypeError(f"Test.channels[{i}] is not a Channel instance.")
        # Normalise tags/meta to base container types
        self.tags = set(self.tags)
        self.meta = dict(self.meta)
        # Derive a sensible name from source_file if name is empty
        if not self.name and self.source_file:
            base = os.path.basename(self.source_file)
            self.name = os.path.splitext(base)[0]

    # ------------------------------------------------------------------ #
    # Basic collection-like behaviour
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """
        Return the number of channels in this test.

        Notes
        -----
        For the full ``(n_channels, n_timesteps)`` shape, use ``test.shape``.
        """
        return len(self.channels)

    def __getitem__(self, key) -> Channel:
        """
        Return a channel by index or name.

        Parameters
        ----------
        key : int or str
            Channel selector:

            * int : position in the channels list (0-based).
            * str : matched against ``Channel.name_user`` or
              ``Channel.name_input`` (case-insensitive exact match).

        Returns
        -------
        Channel
            The selected :class:`Channel` instance.

        Raises
        ------
        KeyError
            If no channel matches a given string key.
        TypeError
            If ``key`` is neither an integer nor a string.
        """
        # Integer index
        if isinstance(key, int):
            return self.channels[key]
        # String key
        if isinstance(key, str):
            key_lower = key.lower()
            # First pass: exact case-insensitive match on name_user
            for ch in self.channels:
                if ch.name_user and ch.name_user.lower() == key_lower:
                    return ch
            # Second pass: exact case-insensitive match on name_input
            for ch in self.channels:
                if ch.name_input and ch.name_input.lower() == key_lower:
                    return ch
            raise KeyError(f"No channel found with name '{key}'")
        raise TypeError(
            f"Test.__getitem__ only supports int or str keys, got {type(key)!r}"
        )

    def iter_channels(
        self,
        selector: ChannelSelector = None,
        tags: Iterable[str] | None = None,
        require_all_tags: bool = False,
    ) -> Iterable[Channel]:
        """
        Iterate over channels selected by index/name and optional tags.

        Parameters
        ----------
        selector : ChannelSelector, optional
            How to pick the initial set of channels:

            * ``None`` (default) : all channels.
            * int : single channel by index.
            * str : single channel by name (via ``__getitem__``).
            * Channel : that channel (if it belongs to this test).
            * slice : slice of the channels list.
            * Sequence[...] : sequence of any of the above.
        tags : iterable of str, optional
            If provided, only channels whose ``ch.tags`` intersect (or
            contain) these will be yielded.
        require_all_tags : bool, optional
            Tag matching rule:

            * ``False`` (default) : keep channel if it has *any* of the
              requested tags.
            * ``True`` : keep channel only if it has *all* requested tags.

        Yields
        ------
        Channel
            Channels matching the selector and tag conditions.

        Raises
        ------
        TypeError
            If ``selector`` has an unsupported type.
        ValueError
            If a :class:`Channel` passed in ``selector`` does not belong
            to this :class:`Test`.
        """
        # Resolve the base set from `selector`
        if selector is None:
            base = list(self.channels)
        elif isinstance(selector, Channel):
            if selector in self.channels:
                base = [selector]
            else:
                raise ValueError("Channel is not part of this Test.")
        elif isinstance(selector, (int, str)):
            base = [self[selector]]
        elif isinstance(selector, slice):
            base = self.channels[selector]
        elif isinstance(selector, Sequence):
            tmp: list[Channel] = []
            for key in selector:
                if isinstance(key, Channel):
                    if key not in self.channels:
                        raise ValueError(
                            "Channel in selector is not part of this Test."
                        )
                    tmp.append(key)
                else:
                    tmp.append(self[key])
            base = tmp
        else:
            raise TypeError(
                f"Unsupported selector type for iter_channels: {type(selector)!r}. "
                "Expected int, str, Channel, slice, Sequence or None."
            )
        # Tag-based filtering
        if tags is None:
            for ch in base:
                yield ch
            return
        required = set(tags)
        for ch in base:
            ch_tags = getattr(ch, "tags", set())
            if not ch_tags:
                continue
            if require_all_tags:
                if required.issubset(ch_tags):
                    yield ch
            else:
                if ch_tags.intersection(required):
                    yield ch

    def channel_names(self) -> list[str]:
        """
        Return a list of preferred channel names for this test.

        For each channel the priority is:

        * ``name_user``
        * ``name_input``
        * fallback: ``"ch{index}"`` (e.g. ``"ch0"``, ``"ch1"``, ...)

        Returns
        -------
        list of str
            List of channel names in the same order as ``self.channels``.
        """
        names: list[str] = []
        for i, ch in enumerate(self.channels):
            if getattr(ch, "name_user", None):
                names.append(ch.name_user)
            elif getattr(ch, "name_input", None):
                names.append(ch.name_input)
            else:
                names.append(f"ch{i}")
        return names

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def channel(self) -> Test:
        """
        Convenience view over this test's channels.

        This property allows access patterns such as::

            test.channel[3]       # 4th Channel (by index)
            test.channel["Acc1"]  # Channel with matching name_input/name_user

        The underlying list is still available as ``test.channels``.
        """
        return self

    @property
    def n_channels(self) -> int:
        """
        Return the number of channels in the test.

        Returns
        -------
        int
            Number of channels.
        """
        return len(self.channels)

    @property
    def n_timesteps(self) -> int:
        """
        Return the number of timesteps (samples) per channel.

        Returns
        -------
        int
            Number of time samples in each channel.

        Raises
        ------
        ValueError
            If channels have differing numbers of samples.
        """
        if not self.channels:
            return 0
        # Length of the first channel's data
        n0 = self.channels[0].data.shape[0]
        # Check all other channels match
        for i, ch in enumerate(self.channels[1:], start=1):
            n_i = ch.data.shape[0]
            if n_i != n0:
                raise ValueError(
                    f"Channels have differing numbers of samples: channel 0 has {n0}, channel {i} has {n_i}"
                )
        return n0

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the ``(n_channels, n_timesteps)`` shape of this test.

        Returns
        -------
        tuple of int
            Tuple ``(n_channels, n_timesteps)``.
        """
        return (self.n_channels, self.n_timesteps)

    @property
    def duration(self) -> float:
        """
        Return the total duration of the test in seconds.

        Returns
        -------
        float
            Total duration of the test.

        Raises
        ------
        ValueError
            If channels have inconsistent durations.
        """
        if not self.channels:
            return 0.0
        # Determine duration of the first channel
        t0, _ = self.channels[0].processed()
        dur0 = t0[-1] - t0[0]
        # Check consistency with all other channels
        for i, ch in enumerate(self.channels[1:], start=1):
            ti, _ = ch.processed()
            dur_i = float(ti[-1] - ti[0])
            if not np.isclose(dur_i, dur0, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"Channels have inconsistent durations: channel 0 duration={dur0}, channel {i} duration={dur_i}"
                )
        return float(dur0)

    @property
    def dt(self) -> float:
        """
        Return the sampling interval of the test in seconds.

        Returns
        -------
        float
            Sampling interval ``dt`` in seconds.

        Raises
        ------
        ValueError
            If the test has no channels or if channels have inconsistent
            ``dt`` values.
        """
        if not self.channels:
            raise ValueError("Cannot determine dt: this Test has no channels.")
        # Reference dt from the first channel
        dt0 = self.channels[0].dt
        # Check consistency with all other channels
        for i, ch in enumerate(self.channels[1:], start=1):
            if not np.isclose(ch.dt, dt0, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"Inconsistent dt across channels: channel 0 dt={dt0}, channel {i} dt={ch.dt}"
                )
        return dt0

    # ------------------------------------------------------------------ #
    # Info / reporting
    # ------------------------------------------------------------------ #

    def info(self) -> str:
        """
        Return a human-readable summary of this test and its channels.

        The summary includes basic test-level metadata, sampling information,
        and a small table of channel-level fields.

        Returns
        -------
        str
            Multi-line string summary (not printed automatically).
        """
        lines: list[str] = []

        # Header (formatted consistently with Channel.info)
        title = self.name or "<unnamed>"
        header = f"Test: {title}"
        lines.append(header)
        lines.append("-" * len(header))

        # Basic test-level metadata
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.source_file:
            lines.append(f"Source file: {self.source_file}")
        if self.timestamp:
            lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Channels: {self.n_channels}")

        # Timesteps
        try:
            n_ts = self.n_timesteps
            lines.append(f"Timesteps: {n_ts}")
        except ValueError as err:
            lines.append(f"Timesteps: <inconsistent> ({err})")

        # Duration
        try:
            dur = self.duration
            lines.append(f"Duration: {dur:.6g} s")
        except ValueError as err:
            lines.append(f"Duration: <inconsistent> ({err})")

        # Sampling
        try:
            dt = self.dt
            fs = float("inf") if dt == 0 else 1.0 / dt
            lines.append(f"Sampling: dt={dt:.6g} s, fs={fs:.6g} Hz")
        except ValueError as err:
            lines.append(f"Sampling: <inconsistent> ({err})")

        # Tags (if any)
        if self.tags:
            lines.append("\nTags:")
            taglist = "\n  ".join(sorted(self.tags))
            lines.append(f"  {taglist}")

        # Free-form metadata
        if self.meta:
            lines.append("\nMetadata:")
            for k, v in self.meta.items():
                lines.append(f"  {k}: {v}")

        # Channel table (blank line before)
        lines.append("")
        # Build table data: rows and headers
        headers = [
            "idx",
            "name_user",
            "name_input",
            "quantity",
            "units",
            "description",
            "tags",
            "calibration_factor",
        ]
        table_rows = []
        for i, ch in enumerate(self.channels):
            table_rows.append(
                [
                    i,
                    ch.name_input or "-",
                    ch.name_user or "-",
                    ch.quantity or "-",
                    ch.units or "-",
                    ch.description_long or "-",
                    ",".join(sorted(ch.tags)) if ch.tags else "-",
                    (
                        f"{ch.calibration_factor:g}"
                        if ch.calibration_factor is not None
                        else "-"
                    ),
                ]
            )
        # Format table
        table_str = tabulate(
            table_rows,
            headers=headers,
            tablefmt="github",
            numalign="right",
            stralign="left",
        )
        lines.append(table_str)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Constructors (I/O)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_channels(
        cls,
        name: str,
        channels: Sequence[Channel],
        description: str | None = None,
        source_file: str | None = None,
        timestamp: str | None = None,
        tags: Iterable[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Test:
        """
        Construct a test directly from an existing sequence of channels.

        Parameters
        ----------
        name : str
            Human-readable name for this test (e.g. ``"Test 07"``).
        channels : sequence of Channel
            Sequence of :class:`Channel` instances to include in the test.
        description : str, optional
            Description of the experiment.
        source_file : str, optional
            Path or identifier of the original data file.
        timestamp : str, optional
            Timestamp string for the experiment.
        tags : iterable of str, optional
            Iterable of test-level tags (e.g. ``{"sofsi", "equals"}``).
        meta : mapping, optional
            Additional metadata (e.g. ``{"specimen_id": "ABC123"}``).

        Returns
        -------
        Test
            New :class:`Test` instance.

        Notes
        -----
        This is a thin convenience wrapper around the dataclass constructor:
        it normalises ``channels``, ``tags`` and ``meta`` into the expected
        container types and lets ``__post_init__`` do the remaining validation.
        """
        # Normalise containers
        channels_list = list(channels)
        tags_set = set(tags) if tags is not None else set()
        meta_dict = dict(meta) if meta is not None else {}
        # Delegate to the regular constructor
        return cls(
            name=name,
            description=description,
            source_file=source_file,
            timestamp=timestamp,
            channels=channels_list,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_sofsi_mat(
        cls,
        filename: str,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Test:
        """
        Construct a test from a SoFSI-style MATLAB ``.mat`` file.

        Expected format
        ---------------
        Required
            ``Channel_1_Data`` : 1D time vector (s)
            ``Channel_i_Data`` : 1D data arrays, ``i >= 2``

        Optional
            ``File_Header`` : struct with any subset of::

                NumberOfChannels, NumberOfSamplesPerChannel,
                SampleFrequency, Date, Comment,
                NumberOfSamplesPerBlock, ...

            ``Channel_i_Header`` : structs with fields::

                SignalName, Unit, MaxLevel, Correction, ...

        Header fields are used only for metadata; missing fields are ignored.

        Parameters
        ----------
        filename : str
            Path to the MAT file.
        name : str, optional
            Test name. If not given, the stem of ``filename`` is used.
        description : str, optional
            Description of the test. If not given, defaults to ``name``.
        tags : iterable of str, optional
            Iterable of test-level tags (e.g. ``{"demo"}``).
        meta : mapping, optional
            Mapping used to initialise ``Test.meta`` (e.g. ``{"specimen": "ABC"}``).

        Returns
        -------
        Test
            New :class:`Test` instance built from the MAT file.

        Raises
        ------
        FileNotFoundError
            If ``filename`` cannot be found.
        KeyError
            If the expected time channel ``"Channel_1_Data"`` is missing.
        ValueError
            If time or data arrays have incompatible shapes.
        """
        try:
            imported_data = sp.io.loadmat(
                filename, squeeze_me=True, struct_as_record=False
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File '{filename}' not found.") from exc

        # Helpers for optional header parsing
        def _parse_int(val):
            try:
                return int(val)
            except Exception:  # noqa: BLE001 - header values are arbitrary MAT objects
                return None

        def _parse_float(val):
            try:
                return float(val)
            except Exception:  # noqa: BLE001 - header values are arbitrary MAT objects
                return None

        def _get_attr(obj, field: str):
            if obj is None:
                return None
            return getattr(obj, field, None)

        # 1. File_Header
        header_dict: dict[str, Any] = {}
        file_header = imported_data.get("File_Header", None)
        if file_header is not None:
            header_dict = {
                "NumberOfChannels": _parse_int(
                    _get_attr(file_header, "NumberOfChannels")
                ),
                "NumberOfSamplesPerChannel": _parse_int(
                    _get_attr(file_header, "NumberOfSamplesPerChannel")
                ),
                "SampleFrequency": _parse_float(
                    _get_attr(file_header, "SampleFrequency")
                ),
                "Date": _get_attr(file_header, "Date"),
                "Comment": _get_attr(file_header, "Comment"),
                "NumberOfSamplesPerBlock": _get_attr(
                    file_header, "NumberOfSamplesPerBlock"
                ),
            }
        # 2. Time vector: Channel_1_Data
        if "Channel_1_Data" not in imported_data:
            raise KeyError(
                "SoFSI MAT file must contain 'Channel_1_Data' as time channel."
            )
        time_vec = np.asarray(imported_data["Channel_1_Data"]).flatten()
        if time_vec.ndim != 1:
            raise ValueError(
                f"'Channel_1_Data' must be 1D (time vector), got shape {time_vec.shape!r}."
            )
        n_samples = len(time_vec)
        # 3. Data channels: Channel_i_Data (i >= 2)
        data_channels_indices: list[int] = []
        for key in imported_data:
            if key.startswith("Channel_") and key.endswith("_Data"):
                try:
                    idx = int(key.split("_")[1])
                except ValueError:
                    continue
                if idx >= 2:
                    data_channels_indices.append(idx)
        if not data_channels_indices:
            raise ValueError(
                "No data channels found in SoFSI MAT file "
                "(no 'Channel_i_Data' with i >= 2)."
            )
        data_channels_indices = sorted(set(data_channels_indices))
        channels: list[Channel] = []
        for idx in data_channels_indices:
            header_key = f"Channel_{idx}_Header"
            data_key = f"Channel_{idx}_Data"
            # Channel header
            hdr = imported_data.get(header_key, None)
            signal_name = _get_attr(hdr, "SignalName") or f"CH{idx-1}"
            unit = _get_attr(hdr, "Unit")
            max_level = _get_attr(hdr, "MaxLevel")
            correction = _get_attr(hdr, "Correction")
            ch_meta: dict[str, Any] = {}
            if max_level is not None:
                parsed_ml = _parse_float(max_level)
                ch_meta["max_level"] = parsed_ml if parsed_ml is not None else max_level
            if correction is not None:
                ch_meta["correction"] = correction
            # Channel data
            data_arr = np.asarray(imported_data[data_key]).flatten()
            if data_arr.ndim != 1:
                raise ValueError(
                    f"'{data_key}' must be 1D (n_samples,), got shape {data_arr.shape!r}."
                )
            if len(data_arr) != n_samples:
                raise ValueError(
                    f"Length of '{data_key}' ({len(data_arr)}) does not match "
                    f"length of Channel_1_Data ({n_samples})."
                )
            # Build Channel
            ch = Channel(
                data=data_arr,
                time=time_vec,
                name_input=str(signal_name),
                units=str(unit).strip() if unit is not None else None,
                raw_units=str(unit).strip() if unit is not None else None,
                meta=ch_meta,
            )
            channels.append(ch)
        if not channels:
            raise ValueError(
                "No data channels found in SoFSI MAT file "
                "(no 'Channel_i_Data' with i >= 2)."
            )
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        timestamp = header_dict.get("Date", None)
        tags_set = set(tags) if tags is not None else set()
        tags_set.add("sofsi")
        meta_dict = dict(meta) if meta is not None else {}
        meta_dict.setdefault("sofsi_file_header", header_dict)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=timestamp,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_equals_mat(
        cls,
        filename: str,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Test:
        """
        Construct a test from an EQUALS-style MATLAB ``.mat`` file.

        Expected format
        ---------------
        Required
            ``t``      : 1D numeric time vector (s)
            ``output`` : 2D numeric array, shape ``(n_samples, n_channels)``

        Optional
            ``Testdate``, ``Time``, ``Frequency``, ``Points``, ``No_Channels``,
            ``File_name``, ``Buffer_Size``, ``sampling``, ``Filter``, ``P_ref``, ...

        Header fields are used only for metadata; missing fields are ignored.

        Parameters
        ----------
        filename : str
            Path to the MAT file.
        name : str, optional
            Test name. If not given, the stem of ``filename`` is used.
        description : str, optional
            Description of the test. If not given, defaults to ``name``.
        tags : iterable of str, optional
            Iterable of test-level tags (e.g. ``{"demo"}``).
        meta : mapping, optional
            Mapping to initialise ``Test.meta`` with
            (e.g. ``{"specimen": "ABC"}``).

        Returns
        -------
        Test
            New :class:`Test` instance built from the MAT file.

        Raises
        ------
        FileNotFoundError
            If ``filename`` cannot be found.
        KeyError
            If required variables ``"t"`` or ``"output"`` are missing.
        ValueError
            If inputs have incompatible shapes.
        """
        try:
            imported_data = sp.io.loadmat(
                filename, squeeze_me=True, struct_as_record=False
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File '{filename}' not found.") from exc

        # Helpers for optional header parsing
        def _get(key: str):
            return imported_data.get(key, None)

        def _parse_int(val):
            try:
                return int(val)
            except Exception:  # noqa: BLE001 - header values are arbitrary MAT objects
                return None

        def _parse_float(val):
            try:
                return float(val)
            except Exception:  # noqa: BLE001 - header values are arbitrary MAT objects
                return None

        # 1. Header metadata
        header_dict = {
            "Testdate": _get("Testdate"),
            "Time": _get("Time"),
            "Frequency": _parse_float(_get("Frequency")),
            "Points": _parse_int(_get("Points")),
            "No_Channels": _parse_int(_get("No_Channels")),
            "File_name": _get("File_name"),
            "Buffer_Size": _parse_int(_get("Buffer_Size")),
            "sampling": _get("sampling"),
            "Filter": _get("Filter"),
            "P_ref": _parse_float(_get("P_ref")),
        }
        # 2. Time vector: t
        if "t" not in imported_data:
            raise KeyError("EQUALS MAT file must contain time vector 't'.")
        time_vec = np.asarray(imported_data["t"]).flatten()
        if time_vec.ndim != 1:
            raise ValueError(f"'t' must be 1D, got shape {time_vec.shape!r}.")
        # 3. Data matrix: output
        if "output" not in imported_data:
            raise KeyError("EQUALS MAT file must contain data matrix 'output'.")
        data_matrix = np.asarray(imported_data["output"])
        if data_matrix.ndim != 2:
            raise ValueError(
                f"'output' must be 2D (n_samples × n_channels), "
                f"got shape {data_matrix.shape!r}."
            )
        n_samples, n_channels = data_matrix.shape
        if len(time_vec) != n_samples:
            raise ValueError(
                f"Time vector length ({len(time_vec)}) does not match "
                f"rows of 'output' ({n_samples})."
            )
        # Build channels
        channels: list[Channel] = []
        for i in range(n_channels):
            ch = Channel(
                data=np.asarray(data_matrix[:, i]).flatten(),
                time=time_vec,
                name_input=f"CH{i+1}",
                name_user=f"CH{i+1}",
            )
            channels.append(ch)
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        testdate = header_dict["Testdate"]
        timestr = header_dict["Time"]
        timestamp: str | None = None
        if isinstance(testdate, str) and isinstance(timestr, str):
            timestamp = f"{testdate} {timestr}"
        elif isinstance(testdate, str):
            timestamp = testdate
        tags_set = set(tags) if tags is not None else set()
        tags_set.add("equals")
        meta_dict = dict(meta) if meta is not None else {}
        meta_dict.setdefault("equals_header", header_dict)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=timestamp,
            tags=tags_set,
            meta=meta_dict,
        )

    @classmethod
    def from_csv(
        cls,
        filename: str,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Test:
        """
        Construct a test from a CSV file in wide format.

        Expected format
        ---------------
        * First row is a header.
        * First column is time (header ``"Time"``).
        * Remaining columns are channels.
        * All values are numeric.

        Parameters
        ----------
        filename : str
            Path to the CSV file.
        name : str, optional
            Test name. If not given, the stem of ``filename`` is used.
        description : str, optional
            Description of the test. If not given, defaults to ``name``.
        tags : iterable of str, optional
            Iterable of test-level tags (e.g. ``{"csv", "demo"}``).
        meta : mapping, optional
            Mapping used to initialise ``Test.meta`` with
            (e.g. ``{"specimen": "ABC"}``).

        Returns
        -------
        Test
            New :class:`Test` instance built from the CSV file.

        Raises
        ------
        ValueError
            If the header is missing or the required time column is not present.
        """
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # must exist
            rows = [row for row in reader]
        if not header:
            raise ValueError("CSV file has no header row.")
        if len(header) < 2:
            raise ValueError("CSV must have Time + at least one data column.")
        if header[0].strip().lower() != "time":
            raise ValueError("First column must be named 'Time'.")
        # Data matrix
        data = np.asarray(rows, dtype=float)
        time_vec = data[:, 0]
        data_cols = data[:, 1:]
        n_samples, n_channels = data_cols.shape
        # Build Channels
        channels: list[Channel] = []
        for i in range(n_channels):
            col_name = header[i + 1].strip() or f"CH{i+1}"
            ch = Channel(
                data=data_cols[:, i],
                time=time_vec,
                name_input=col_name,
                name_user=col_name,
            )
            channels.append(ch)
        # Build Test
        if not name:
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
        tags_set = set(tags) if tags else set()
        tags_set.add("csv")
        meta_dict = dict(meta) if meta else {}
        meta_dict.setdefault("csv_header", header)
        meta_dict.setdefault("csv_n_samples", n_samples)
        meta_dict.setdefault("csv_n_channels", n_channels)
        return cls.from_channels(
            name=name,
            channels=channels,
            description=description,
            source_file=filename,
            timestamp=None,
            tags=tags_set,
            meta=meta_dict,
        )

    def to_csv(
        self,
        filename: str,
        selector: ChannelSelector = None,
        include_axis_labels: bool = True,
        overwrite: bool = True,
    ) -> None:
        """
        Export selected channels to a CSV file (wide format).

        The output has the form::

            Time, Ch1, Ch2, ...

        Parameters
        ----------
        filename : str
            Output CSV file path.
        selector : ChannelSelector, optional
            Channels to export (index, name, list, slice, etc.).
            If ``None`` (default), all channels are exported.
        include_axis_labels : bool, optional
            If ``True`` (default), use ``channel.label_axis`` (preferred),
            ``channel.name_user`` or ``channel.name_input`` as column names.
            Otherwise fallback to ``"Ch1"``, ``"Ch2"``, ...
        overwrite : bool, optional
            If ``False`` and the file already exists, a
            :class:`FileExistsError` is raised.

        Raises
        ------
        FileExistsError
            If ``overwrite`` is ``False`` and the file exists.
        ValueError
            If no channels are selected or if selected channels do not share
            the same time vector.
        """
        # Check overwrite
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        # Resolve channels
        selected_channels = list(self.iter_channels(selector))
        if not selected_channels:
            raise ValueError("No channels selected for CSV export.")
        # Time vector
        t0 = selected_channels[0].time
        for ch in selected_channels[1:]:
            if not np.array_equal(ch.time, t0):
                raise ValueError(
                    "All selected channels must share the same time vector "
                    "to export as a single CSV table."
                )
        # Build headers
        headers: list[str] = ["Time"]
        for i, ch in enumerate(selected_channels):
            if include_axis_labels:
                label = ch.label_axis or ch.name_user or ch.name_input or f"Ch{i+1}"
                headers.append(label)
            else:
                headers.append(f"Ch{i+1}")
        # Build rows
        data_matrix = np.column_stack([t0] + [ch.data for ch in selected_channels])
        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data_matrix)

    # ------------------------------------------------------------------ #
    # CSV utilities
    # ------------------------------------------------------------------ #

    def channel_info_to_csv(
        self,
        filename: str,
        overwrite: bool = True,
    ) -> None:
        """
        Export channel metadata to CSV (one row per channel).

        Columns are written only if they contain non-identical data
        across channels, to avoid redundant storage.

        Notes
        -----
        The column ``"idx"`` (channel index) is always included.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        overwrite : bool, optional
            If ``False`` and the file exists, a :class:`FileExistsError`
            is raised. Default is ``True``.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        # Fields we consider for export
        fields = [
            "name_user",
            "name_input",
            "quantity",
            "units",
            "raw_units",
            "label_axis",
            "label_legend",
            "description_long",
            "calibration_factor",
            "tags",
        ]

        # Priority list for intra-channel deduplication
        priority = ["name_input", "name_user", "label_axis", "label_legend"]

        # Build per-channel value dicts. For string fields we normalise (strip)
        # and for others (numbers, tags) keep original values.
        per_ch_vals: list[dict[str, Any]] = []
        for ch in self.channels:
            v: dict[str, Any] = {}
            for f in fields:
                raw = getattr(ch, f, None)
                if f in ("name_input", "name_user", "label_axis", "label_legend"):
                    if raw is None:
                        v[f] = None
                    else:
                        s = str(raw).strip()
                        v[f] = s if s != "" else None
                elif f == "tags":
                    v[f] = set(raw) if raw else set()
                else:
                    v[f] = raw
            per_ch_vals.append(v)

        # Apply intra-channel suppression according to priority:
        # if a lower-priority value equals any kept higher-priority value for same channel -> drop it.
        for v in per_ch_vals:
            seen: set[str] = set()
            for p in priority:
                val = v.get(p)
                if val is None:
                    continue
                # Compare using exact (stripped) equality
                if val in seen:
                    v[p] = None
                else:
                    seen.add(val)

        # Decide which fields actually have at least one non-empty value after suppression
        fields_to_export = ["idx"]
        for f in fields:
            if f == "tags":
                # Include tags column if any channel has non-empty tags
                any_nonempty = any(bool(v["tags"]) for v in per_ch_vals)
            elif f in ("name_input", "name_user", "label_axis", "label_legend"):
                any_nonempty = any(v.get(f) not in (None, "") for v in per_ch_vals)
            else:
                # For other fields include if any channel has a non-None / non-empty value
                any_nonempty = any(
                    v.get(f) is not None and v.get(f) != "" for v in per_ch_vals
                )
            if any_nonempty:
                fields_to_export.append(f)

        # Write CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields_to_export)
            for idx, ch in enumerate(self.channels):
                row = []
                vals = per_ch_vals[idx]
                for field in fields_to_export:
                    if field == "idx":
                        row.append(idx)
                    elif field == "tags":
                        row.append(
                            ",".join(sorted(vals["tags"])) if vals["tags"] else ""
                        )
                    else:
                        value = vals.get(field)
                        # Preserve numeric types (e.g. calibration_factor). Convert others to string.
                        if value is None:
                            row.append("")
                        else:
                            if isinstance(value, (int, float)):
                                row.append(value)
                            else:
                                row.append(value)
                writer.writerow(row)

    def with_channel_info_from_csv(self, filename: str) -> Test:
        """
        Return a new test with channel metadata updated from CSV.

        Header matching is flexible (underscores/spaces and case are ignored),
        and rows are matched to channels as follows:

        1. If an ``idx`` column is present, it is used first.
        2. Otherwise, matching is attempted against ``name_user`` or
           ``name_input`` (case-insensitive).

        Any row that cannot be matched raises a :class:`ValueError`.
        Only metadata is updated; data and processing parameters are
        left unchanged.

        Parameters
        ----------
        filename : str
            Path to the CSV file produced e.g. by ``channel_info_to_csv``.

        Returns
        -------
        Test
            New :class:`Test` instance with updated channel metadata.

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.
        ValueError
            If the CSV is empty, has no recognised columns, or a row cannot
            be matched to any channel.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        # Load CSV
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty.")
            rows = list(reader)
        if not header:
            raise ValueError("CSV has no header.")

        # Normaliser for column names
        def norm(s: str) -> str:
            return s.strip().lower().replace(" ", "_")

        # Alias mapping (all lower-case and underscore-normalised)
        alias_map = {
            "idx": {"idx", "index", "#"},
            "name_user": {"name_user", "user", "user_name", "name", "channel", "ch"},
            "name_input": {"name_input", "input", "input_name", "raw_name", "daq"},
            "quantity": {"quantity", "qty"},
            "units": {"units", "unit"},
            "raw_units": {"raw_units", "rawunit"},
            "label_axis": {"label_axis", "axis_label", "ylabel", "y_label"},
            "label_legend": {"label_legend", "legend_label", "legend"},
            "description_long": {"description_long", "description", "desc"},
            "calibration_factor": {
                "calibration_factor",
                "calibration",
                "calib",
                "gain",
            },
            "tags": {"tags", "tag"},
        }
        # Build column to attribute map
        col_to_attr = {}
        for i, name in enumerate(header):
            key = norm(name)
            for attr, aliases in alias_map.items():
                if key in aliases:
                    col_to_attr[i] = attr
                    break
        if not col_to_attr:
            raise ValueError("No recognised columns in CSV header.")
        # Pre-build name lookup (case-insensitive)
        name_user_map = {
            (ch.name_user or "").lower(): i
            for i, ch in enumerate(self.channels)
            if ch.name_user
        }
        name_input_map = {
            (ch.name_input or "").lower(): i
            for i, ch in enumerate(self.channels)
            if ch.name_input
        }
        idx_col = next((i for i, a in col_to_attr.items() if a == "idx"), None)
        new_channels = list(self.channels)

        # Precompute whether CSV provides explicit label columns
        has_label_axis_col = any(a == "label_axis" for a in col_to_attr.values())
        has_label_legend_col = any(a == "label_legend" for a in col_to_attr.values())

        for row in rows:
            if not any(row):
                continue
            ch_idx = None
            # Attempt 1: match by idx column
            if idx_col is not None and idx_col < len(row):
                try:
                    ix = int(row[idx_col])
                    if 0 <= ix < len(self.channels):
                        ch_idx = ix
                except ValueError:
                    pass
            # Attempt 2: match by name
            if ch_idx is None:
                names = []
                for col, attr in col_to_attr.items():
                    if attr in ("name_user", "name_input") and col < len(row):
                        v = row[col].strip()
                        if v:
                            names.append(v.lower())
                for nm in names:
                    if nm in name_user_map:
                        ch_idx = name_user_map[nm]
                        break
                    if nm in name_input_map:
                        ch_idx = name_input_map[nm]
                        break
            if ch_idx is None:
                raise ValueError(f"Row could not match any channel: {row}")

            ch = new_channels[ch_idx]

            # Collect raw string values for all recognised attributes in this row
            row_vals: dict[str, str] = {}
            for col, attr in col_to_attr.items():
                if col < len(row):
                    row_vals[attr] = row[col].strip()
                else:
                    row_vals[attr] = ""

            updates: dict[str, Any] = {}
            tags_val = None

            # Handle tags if present
            if row_vals.get("tags"):
                tokens = [
                    t.strip()
                    for t in row_vals["tags"].replace(";", ",").split(",")
                    if t.strip()
                ]
                tags_val = set(tokens)

            # calibration_factor parsing
            cf_raw = row_vals.get("calibration_factor", "")
            if cf_raw:
                try:
                    updates["calibration_factor"] = float(cf_raw)
                except ValueError:
                    # ignore bad numeric conversion, keep existing calibration
                    pass

            # If either name_user OR name_input is provided in this row, treat it
            # as a renaming operation: let the Channel constructor regenerate
            # default labels (name_user falls back to name_input inside Channel.__post_init__).
            name_user_provided = bool(row_vals.get("name_user"))
            name_input_provided = bool(row_vals.get("name_input"))
            name_key_provided = name_user_provided or name_input_provided

            # Copy other string-like fields. For label_axis/label_legend:
            # - If CSV provides a non-empty value, use it.
            # - Else if name_user_provided, set explicit None so Channel.__post_init__
            #   will create sensible defaults (possibly using units).
            for attr in (
                "name_input",
                "name_user",
                "quantity",
                "units",
                "raw_units",
                "description_long",
            ):
                val = row_vals.get(attr)
                if val:
                    updates[attr] = val

            # label_axis
            la_val = row_vals.get("label_axis", "")
            if la_val:
                updates["label_axis"] = la_val
            else:
                # If either name_user or name_input was provided and no explicit
                # label_axis was supplied in the CSV, set None so Channel.__post_init__
                # can create sensible defaults (name_user preferred, else name_input).
                if (name_key_provided and not has_label_axis_col) or (
                    name_key_provided and has_label_axis_col and la_val == ""
                ):
                    updates["label_axis"] = None

            # label_legend
            ll_val = row_vals.get("label_legend", "")
            if ll_val:
                updates["label_legend"] = ll_val
            else:
                if (name_key_provided and not has_label_legend_col) or (
                    name_key_provided and has_label_legend_col and ll_val == ""
                ):
                    updates["label_legend"] = None

            if tags_val is not None:
                updates["tags"] = tags_val

            # Preserve numeric types and pass through strings; replace() will call __post_init__
            if updates:
                new_channels[ch_idx] = replace(ch, **updates)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Channel management
    # ------------------------------------------------------------------ #

    def add_channel(self, ch: Channel) -> Test:
        """
        Return a new test with a channel appended to the channel list.

        The original :class:`Test` instance is left unchanged.

        Parameters
        ----------
        ch : Channel
            Channel instance to append.

        Returns
        -------
        Test
            New :class:`Test` instance with one extra channel.

        Raises
        ------
        TypeError
            If ``ch`` is not a :class:`Channel` instance.
        """
        if not isinstance(ch, Channel):
            raise TypeError("add_channel expects a Channel instance.")
        new_channels = list(self.channels)
        new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def drop_channels(self, selector: ChannelSelector) -> Test:
        """
        Return a new test with the selected channels removed.

        Parameters
        ----------
        selector : ChannelSelector
            Channels to remove. Can be anything accepted by
            :meth:`iter_channels`, e.g.:

            * int : index.
            * str : name (``name_user`` / ``name_input``).
            * Channel : that channel.
            * slice : slice of the channels list.
            * Sequence[...] : list of the above.

        Returns
        -------
        Test
            New :class:`Test` instance with selected channels removed.

        Raises
        ------
        ValueError
            If ``selector`` is ``None``.
        """
        if selector is None:
            raise ValueError("drop_channels requires a non-None selector.")
        # Resolve which Channel objects to remove
        channels_to_drop = list(self.iter_channels(selector))
        # Keep all channels that are not in channels_to_drop
        new_channels = [ch for ch in self.channels if ch not in channels_to_drop]
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Batch processing
    # ------------------------------------------------------------------ #

    def drift_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> Test:
        """
        Return a new test with drift-corrected versions of selected channels.

        This applies :meth:`Channel.drift_corrected` to the selected
        channels and leaves the others unchanged.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to process (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are processed.
        **override
            Keyword arguments forwarded to :meth:`Channel.drift_corrected`,
            e.g. ``points=100``. These override stored drift parameters.

        Returns
        -------
        Test
            New :class:`Test` instance with updated channels.

        Raises
        ------
        ValueError
            If no channels are selected for drift correction.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for drift correction.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.drift_corrected(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def filtered(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> Test:
        """
        Return a new test with filtered versions of selected channels.

        This applies :meth:`Channel.filtered` to the selected channels.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to process (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are processed.
        **override
            Keyword arguments forwarded to :meth:`Channel.filtered`, e.g.
            ``btype="highpass"``, ``fc=0.5``, ``order=4``. These override
            the stored filter parameters.

        Returns
        -------
        Test
            New :class:`Test` instance with updated channels.

        Raises
        ------
        ValueError
            If no channels are selected for filtering.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for filtering.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.filtered(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def baseline_corrected(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> Test:
        """
        Return a new test with baseline-corrected selected channels.

        This applies :meth:`Channel.baseline_corrected` to the selected
        channels.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to process (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are processed.
        **override
            Keyword arguments forwarded to :meth:`Channel.baseline_corrected`,
            e.g. ``type="linear"``. These override stored baseline parameters.

        Returns
        -------
        Test
            New :class:`Test` instance with updated channels.

        Raises
        ------
        ValueError
            If no channels are selected for baseline correction.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for baseline correction.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.baseline_corrected(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed(
        self,
        selector: ChannelSelector = None,
        **override: Any,
    ) -> Test:
        """
        Return a new test with manually trimmed selected channels.

        This is the generic manual-window trimming interface based on
        explicit ``t_start`` and ``t_end`` (in seconds), using
        :meth:`Channel.trimmed`.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to process (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are processed.
        **override
            Keyword arguments forwarded to :meth:`Channel.trimmed`, typically
            including ``t_start`` and ``t_end`` (in seconds).

        Returns
        -------
        Test
            New :class:`Test` instance with updated channels.

        Raises
        ------
        ValueError
            If no channels are selected for trimming.
        """
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for trimming.")
        new_channels: list[Channel] = []
        for ch in self.channels:
            if ch in selected:
                new_channels.append(ch.trimmed(**override))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_threshold(
        self,
        selector: ChannelSelector = None,
        ref: ChannelKey | None = None,
        threshold: float = 0.01,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> Test:
        """
        Return a new test with threshold-based aligned trimming.

        A single time window is derived from one reference channel using
        :meth:`Channel.trim_by_threshold`, and then applied to all
        selected channels.

        Strategy
        --------
        * Choose a reference channel:

          - If ``ref`` is given, use that (must belong to this test and
            be part of the selected set).
          - Otherwise, use the first selected channel.

        * On the reference channel, compute a threshold-based window via
          :meth:`Channel.trim_by_threshold`.
        * Extract ``t_start`` and ``t_end`` from the reference channel's
          ``trim_params``.
        * Apply :meth:`Channel.trimmed` with this window to all selected
          channels.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to trim (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are trimmed.
        ref : ChannelKey, optional
            Reference channel used to define the trim window. Can be an
            index, name, or :class:`Channel` instance. If ``None``, the
            first selected channel is used. The reference must be part of
            the selected set.
        threshold : float, optional
            Threshold value in signal units used to detect when the motion
            starts/stops (see :meth:`Channel.trim_by_threshold`).
        use_abs : bool, optional
            If ``True`` (default), thresholding is applied to
            ``abs(signal)``. If ``False``, thresholding is applied to the
            raw signal.
        buffer_before : float, optional
            Time buffer (in seconds) to extend the window before the
            detected start time.
        buffer_after : float, optional
            Time buffer (in seconds) to extend the window after the
            detected end time.
        processed : bool, optional
            Whether to use processed data for the reference channel when
            computing the window. Default is ``True``.
        use_cache : bool, optional
            Whether to use the channel-level processing cache for the
            reference channel. Default is ``True``.

        Returns
        -------
        Test
            New :class:`Test` instance with aligned trimming across channels.

        Raises
        ------
        ValueError
            If no channels are selected, if the reference channel is not
            part of this test, or if it is not in the selected set.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for threshold-based trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]  # int or str via __getitem__
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_threshold(
            threshold=threshold,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_fraction_of_peak(
        self,
        selector: ChannelSelector = None,
        ref: ChannelKey | None = None,
        fraction: float = 0.05,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> Test:
        """
        Return a new test with fraction-of-peak aligned trimming.

        A single time window is derived from a fraction-of-peak criterion
        on one reference channel using
        :meth:`Channel.trim_by_fraction_of_peak`, and then applied to all
        selected channels.

        Strategy
        --------
        * Choose a reference channel (as in
          :meth:`trimmed_by_threshold`).
        * On the reference channel, compute the window via
          :meth:`Channel.trim_by_fraction_of_peak`.
        * Extract ``t_start`` and ``t_end`` from the reference channel's
          ``trim_params``.
        * Apply :meth:`Channel.trimmed` with this window to all selected
          channels.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to trim (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are trimmed.
        ref : ChannelKey, optional
            Reference channel used to define the trim window. Can be an
            index, name, or :class:`Channel` instance. If ``None``, the
            first selected channel is used. The reference must be part of
            the selected set.
        fraction : float, optional
            Fraction of the peak amplitude in ``(0, 1]`` used to define the
            effective-motion window (see
            :meth:`Channel.trim_by_fraction_of_peak`).
        use_abs : bool, optional
            If ``True`` (default), use absolute amplitude when computing the
            peak.
        buffer_before : float, optional
            Time buffer (in seconds) to extend the window before the
            detected start time.
        buffer_after : float, optional
            Time buffer (in seconds) to extend the window after the
            detected end time.
        processed : bool, optional
            Whether to use processed data for the reference channel when
            computing the window.
        use_cache : bool, optional
            Whether to use the channel-level processing cache for the
            reference channel.

        Returns
        -------
        Test
            New :class:`Test` instance with aligned trimming across channels.

        Raises
        ------
        ValueError
            If no channels are selected, if the reference channel is not
            part of this test, or if it is not in the selected set.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for fraction-of-peak trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_fraction_of_peak(
            fraction=fraction,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    def trimmed_by_arias(
        self,
        selector: ChannelSelector = None,
        ref: ChannelKey | None = None,
        lower: float = 0.05,
        upper: float = 0.95,
        g: float = 9.81,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> Test:
        """
        Return a new test with Arias-intensity-based aligned trimming.

        A single significant-duration window is derived from one reference
        channel using :meth:`Channel.trim_by_arias`, and then applied to
        all selected channels.

        Strategy
        --------
        * Choose a reference channel (as in
          :meth:`trimmed_by_threshold`).
        * On the reference channel, compute the Arias-based window via
            :meth:`Channel.trim_by_arias`.
        * Extract ``t_start`` and ``t_end`` from the reference channel's
          ``trim_params``.
        * Apply :meth:`Channel.trimmed` with this window to all selected
          channels.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to trim (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are trimmed.
        ref : ChannelKey, optional
            Reference channel used to define the trim window. Can be an
            index, name, or :class:`Channel` instance. If ``None``, the
            first selected channel is used. The reference must be part of
            the selected set.
        lower : float, optional
            Lower fraction of Arias intensity in ``[0, 1]`` that defines
            the start of the significant-duration window (typically 0.05).
        upper : float, optional
            Upper fraction of Arias intensity in ``[0, 1]`` that defines
            the end of the significant-duration window (typically 0.95).
        g : float, optional
            Gravitational acceleration in m/s² used for Arias intensity.
        buffer_before : float, optional
            Time buffer (in seconds) to extend the window before the
            detected lower time.
        buffer_after : float, optional
            Time buffer (in seconds) to extend the window after the
            detected upper time.
        processed : bool, optional
            Whether to use processed data for the reference channel when
            computing the window.
        use_cache : bool, optional
            Whether to use the channel-level processing cache for the
            reference channel.

        Returns
        -------
        Test
            New :class:`Test` instance with aligned trimming across channels.

        Raises
        ------
        ValueError
            If no channels are selected, if the reference channel is not
            part of this test, or if it is not in the selected set.
        """
        # Resolve selected channels
        if selector is None:
            selected = list(self.channels)
        else:
            selected = list(self.iter_channels(selector))
        if not selected:
            raise ValueError("No channels selected for Arias-based trimming.")
        # Resolve reference channel
        if ref is None:
            ref_ch = selected[0]
        else:
            if isinstance(ref, Channel):
                if ref not in self.channels:
                    raise ValueError("Reference channel is not part of this Test.")
                ref_ch = ref
            else:
                ref_ch = self[ref]
        if ref_ch not in selected:
            raise ValueError("Reference channel must be part of the selected channels.")
        # Compute window on reference channel
        ref_trimmed = ref_ch.trim_by_arias(
            lower=lower,
            upper=upper,
            g=g,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )
        params = getattr(ref_trimmed, "trim_params", {})
        t_start = params.get("t_start", float(ref_ch.time[0]))
        t_end = params.get("t_end", float(ref_ch.time[-1]))
        # Apply same window to all selected channels
        new_channels: list[Channel] = []
        for ch in self.channels:
            if any(ch is s for s in selected):
                new_channels.append(ch.trimmed(t_start=t_start, t_end=t_end))
            else:
                new_channels.append(ch)
        return type(self)(
            name=self.name,
            description=self.description,
            source_file=self.source_file,
            timestamp=self.timestamp,
            channels=new_channels,
            tags=set(self.tags),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Pairwise analysis
    # ------------------------------------------------------------------ #

    def cross_spectrum(
        self,
        x: ChannelKey,
        y: ChannelKey,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the cross power spectral density (CPSD) between two channels.

        The CPSD is computed using :func:`scipy.signal.csd`.

        Parameters
        ----------
        x : ChannelKey
            Input (excitation) channel key (index, name or :class:`Channel`).
        y : ChannelKey
            Output (response) channel key (index, name or :class:`Channel`).
        processed : bool, optional
            If ``True`` (default), use processed data from each channel.
        use_cache : bool, optional
            If ``True`` (default), use the channel-level processing cache.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`scipy.signal.csd`, e.g. ``nperseg``, ``window``,
            ``noverlap``. If ``nperseg`` is not given, a MATLAB-like
            default of ``min(256, n)`` is used.

        Returns
        -------
        f : np.ndarray
            Frequency array in Hz.
        Pxy : np.ndarray
            Complex cross-spectrum :math:`P_{xy}(f)`.

        Raises
        ------
        ValueError
            If the channels are empty, have different lengths, or have
            inconsistent or non-positive ``dt`` values.
        """
        if isinstance(x, Channel):
            if not any(x is s for s in self.channels):
                raise ValueError("Input channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if not any(y is s for s in self.channels):
                raise ValueError("Output channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute cross spectrum of empty signal.")
        if x_data.size != y_data.size:
            raise ValueError("Channels must have the same length for cross spectrum.")
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Cross spectrum requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Cross spectrum requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        fs = 1.0 / ch_x.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, Pxy = sp.signal.csd(x_data, y_data, fs=fs, **kwargs)
        return f, Pxy

    def transfer_function(
        self,
        x: ChannelKey,
        y: ChannelKey,
        kind: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the frequency-domain transfer function between two channels.

        Cross- and auto-spectra are computed with
        :func:`scipy.signal.csd`. Two standard estimators are supported:

        * H1: ``H1(f) = G_yx(f) / G_xx(f)``, preferred when the input is noisy.
        * H2: ``H2(f) = G_yy(f) / G_yx(f)``, preferred when the output is noisy.

        Here, ``G_yx`` is the cross-spectrum between output ``y`` and input
        ``x``, and ``G_xx`` / ``G_yy`` are the auto-spectra of input and
        output.

        Parameters
        ----------
        x : ChannelKey
            Input (excitation) channel key (index, name or :class:`Channel`).
        y : ChannelKey
            Output (response) channel key (index, name or :class:`Channel`).
        kind : {"H1", "H2"}, optional
            Type of transfer-function estimator (default ``"H1"``).
        processed : bool, optional
            If ``True`` (default), use processed data from each channel.
        use_cache : bool, optional
            If ``True`` (default), use the channel-level processing cache.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`scipy.signal.csd`, e.g. ``nperseg``, ``window``, ``noverlap``.
            If ``nperseg`` is not given, a MATLAB-like default of ``min(256, n)``
            is used.

        Returns
        -------
        f : np.ndarray
            Frequency array in Hz.
        H : np.ndarray
            Complex transfer function values :math:`H(f)`.

        Raises
        ------
        ValueError
            If the channels are empty, have different lengths, or have

            inconsistent or non-positive ``dt`` values, or if ``kind`` is
            not one of ``"H1"`` or ``"H2"``.
        """
        if isinstance(x, Channel):
            if not any(x is s for s in self.channels):
                raise ValueError("Input channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if not any(y is s for s in self.channels):
                raise ValueError("Output channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute transfer function of empty signal.")
        if x_data.size != y_data.size:
            raise ValueError(
                "Channels must have the same length for transfer function."
            )
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Transfer function requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Transfer function requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        fs = 1.0 / ch_x.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, Gyx = sp.signal.csd(y_data, x_data, fs=fs, **kwargs)
        _, Gxx = sp.signal.csd(x_data, x_data, fs=fs, **kwargs)
        _, Gyy = sp.signal.csd(y_data, y_data, fs=fs, **kwargs)
        kind_u = kind.upper()
        if kind_u == "H1":
            H = Gyx / Gxx
        elif kind_u == "H2":
            H = Gyy / Gyx
        else:
            raise ValueError(
                f"Unsupported transfer-function kind {kind!r}, use 'H1' or 'H2'."
            )
        return f, H

    def plot_transfer_function(
        self,
        x: ChannelKey,
        y: ChannelKey,
        kind: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        phase: bool = False,
        fmax: float | None = None,
        ax: plt.Axes | None = None,
        tf_kwargs: Mapping[str, Any] | None = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the magnitude (and optionally phase) of the transfer function.

        This is a convenience wrapper around :meth:`transfer_function`:

        * Computes H1 or H2 transfer function between input ``x`` and output
          ``y``.
        * Plots the linear magnitude ``|H(f)|`` versus frequency on a
          logarithmic x-axis.
        * Optionally overlays the phase angle in degrees on a secondary
          y-axis.

        Parameters
        ----------
        x : ChannelKey
            Input (excitation) channel key (index, name or :class:`Channel`).
        y : ChannelKey
            Output (response) channel key (index, name or :class:`Channel`).
        kind : {"H1", "H2"}, optional
            Transfer function estimator (default ``"H1"``).
        processed : bool, optional
            If ``True`` (default), use processed data from each channel.
        use_cache : bool, optional
            If ``True`` (default), use the channel-level processing cache.
        phase : bool, optional
            If ``True``, also plot the phase angle (in degrees) on a
            secondary y-axis.
        fmax : float or None, optional
            Optional upper frequency limit in Hz for plotting. If ``None``,
            the full available frequency range is plotted.
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        tf_kwargs : mapping or None, optional
            Additional keyword arguments forwarded to
            :meth:`transfer_function` and ultimately :func:`scipy.signal.csd`,
            e.g. ``nperseg``, ``window``, ``noverlap``.
        label : str or None, optional
            Label for the magnitude curve (for legends). If ``None``, a
            default label based on the channel names is used.
        **plot_kwargs
            Additional keyword arguments forwarded to :meth:`Axes.semilogx`
            (e.g. ``linestyle``, ``linewidth``).

        Returns
        -------
        matplotlib.axes.Axes
            Axes with the plotted transfer function magnitude (and,
            optionally, phase).
        """
        if tf_kwargs is None:
            tf_kwargs = {}

        # Compute transfer function (will validate channels and sampling)
        f, H = self.transfer_function(
            x=x,
            y=y,
            kind=kind,
            processed=processed,
            use_cache=use_cache,
            **tf_kwargs,
        )

        mag = np.abs(H)

        # Apply frequency limit if requested
        if fmax is not None:
            mask = f <= fmax
            f = f[mask]
            mag = mag[mask]
            H = H[mask]

        if ax is None:
            _, ax = plt.subplots()

        # Build a sensible default label if none provided
        if label is None:
            ch_x = x if isinstance(x, Channel) else self[x]
            ch_y = y if isinstance(y, Channel) else self[y]
            name_x = ch_x.name_user or ch_x.name_input or "<x>"
            name_y = ch_y.name_user or ch_y.name_input or "<y>"
            label = f"{kind.upper()} {name_y}/{name_x}"

        # Plot magnitude
        ax.semilogx(f, mag, label=label, **plot_kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"Transfer function magnitude $|H(f)|$")
        ax.grid(True, which="both", linestyle=":")

        # Optionally add phase on secondary axis
        if phase:
            # Unwrap phase to avoid ±π jumps, then convert to degrees
            phase_deg = np.unwrap(np.angle(H)) * 180.0 / np.pi
            ax_phase = ax.twinx()
            ax_phase.semilogx(f, phase_deg, linestyle="--")
            ax_phase.set_ylabel("Phase [deg]")

        return ax

    def time_delay(
        self,
        x: ChannelKey,
        y: ChannelKey,
        processed: bool = True,
        use_cache: bool = True,
    ) -> float:
        """
        Estimate the time delay between two channels using cross-correlation.

        A positive delay means that the output ``y`` lags the input ``x``,
        based on the lag at which the cross-correlation between ``y`` and
        ``x`` is maximised.

        Parameters
        ----------
        x : ChannelKey
            Input (excitation) channel key (index, name or :class:`Channel`).
        y : ChannelKey
            Output (response) channel key (index, name or :class:`Channel`).
        processed : bool, optional
            If ``True`` (default), use processed data from each channel.
        use_cache : bool, optional
            If ``True`` (default), use the channel-level processing cache.

        Returns
        -------
        float
            Estimated time delay in seconds (positive if ``y`` lags ``x``).

        Raises
        ------
        ValueError
            If the channels are empty, have different lengths, or have
            inconsistent or non-positive ``dt`` values.
        """
        if isinstance(x, Channel):
            if not any(x is s for s in self.channels):
                raise ValueError("Input channel is not part of this Test.")
            ch_x = x
        else:
            ch_x = self[x]
        if isinstance(y, Channel):
            if not any(y is s for s in self.channels):
                raise ValueError("Output channel is not part of this Test.")
            ch_y = y
        else:
            ch_y = self[y]
        _, x_data = ch_x.xy(processed=processed, use_cache=use_cache)
        _, y_data = ch_y.xy(processed=processed, use_cache=use_cache)
        if x_data.size == 0 or y_data.size == 0:
            raise ValueError("Cannot compute time delay for empty signal.")
        if x_data.size != y_data.size:
            raise ValueError(
                "Channels must have the same length for time-delay estimation."
            )
        if ch_x.dt is None or ch_x.dt <= 0.0:
            raise ValueError(
                "Time-delay estimation requires a positive dt on the input channel."
            )
        if ch_y.dt is None or ch_y.dt <= 0.0:
            raise ValueError(
                "Time-delay estimation requires a positive dt on the output channel."
            )
        if ch_x.dt != ch_y.dt:
            raise ValueError(
                "Input and output channels must have the same sampling interval dt."
            )
        n = x_data.size
        x0 = x_data - float(np.mean(x_data))
        y0 = y_data - float(np.mean(y_data))
        c = np.correlate(y0, x0, mode="full")
        lags = np.arange(-n + 1, n)
        k = lags[int(np.argmax(c))]
        return float(k * ch_x.dt)

    # ------------------------------------------------------------------ #
    # Basic modal identification (skeleton only)
    # ------------------------------------------------------------------ #

    def ema_model(
        self,
        input: ChannelKey,
        outputs: ChannelSelector,
        kind: Literal["H1", "H2"] = "H1",
        processed: bool = True,
        use_cache: bool = True,
        **model_kwargs: Any,
    ):
        """
        Build and return an ``sdypy.EMA.Model`` for experimental modal analysis.

        This method computes frequency-response functions (FRFs) between one
        input channel and multiple output channels using
        :meth:`transfer_function`, and then constructs an
        ``sdypy.EMA.Model`` instance using ``**model_kwargs``.

        The returned object provides pole estimation, stabilisation charts,
        modal parameter extraction and FRF reconstruction.

        Parameters
        ----------
        input : ChannelKey
            Input (excitation) channel.
        outputs : ChannelSelector
            Output (response) channels (single selector or sequence).
        kind : {"H1", "H2"}, optional
            Transfer function estimator (default ``"H1"``).
        processed : bool, optional
            Use processed channel data (default ``True``).
        use_cache : bool, optional
            Use channel-level cache (default ``True``).
        **model_kwargs
            Additional keyword arguments passed directly to
            ``sdypy.EMA.Model``. Typical options include:

            * ``lower`` : lower frequency for pole estimation.
            * ``upper`` : upper frequency for pole estimation.
            * ``pol_order_high`` : highest model order for LSCF.
            * ``driving_point`` : index of driving FRF.
            * ``frf_type`` : ``"accelerance"``, ``"mobility"``,
              ``"receptance"``, etc.

        Returns
        -------
        sdypy.EMA.Model
            Modal analysis model from the ``sdypy.EMA`` package.

        Raises
        ------
        ImportError
            If the optional dependency ``sdypy`` is not installed.
        ValueError
            If no output channels are selected or if FRFs do not share
            identical frequency grids.

        Examples
        --------
        Typical usage::

            model = test.ema_model(input="Shaker", outputs=[...], lower=1.0, upper=50.0)

            # 1) Get poles (LSCF)
            model.get_poles()

            # 2) Select stable poles (interactive or automatic)
            model.select_poles()
            # or
            model.select_closest_poles([f1, f2, ...])

            # 3) Print modal data (natural frequencies, damping, mode shapes)
            model.print_modal_data()
            # or
            print(model.nat_freq)
            print(model.nat_xi)
            print(model.phi)

            # 4) Reconstruct FRFs and modal constants
            frf_rec, modal_const = model.get_constants()
        """
        try:
            from sdypy import EMA
        except ImportError as exc:
            raise ImportError(
                "Experimental modal analysis requires the optional dependency "
                "'sdypy'. Install it with 'pip install sdypy-EMA'."
            ) from exc
        # Resolve channels
        if isinstance(input, Channel):
            ch_in = input
        else:
            ch_in = self[input]
        outs = list(self.iter_channels(outputs))
        if not outs:
            raise ValueError("No output channels selected.")
        # Compute FRFs for each output channel
        f_ref = None
        H_rows = []
        for ch_out in outs:
            f, H = self.transfer_function(
                x=ch_in,
                y=ch_out,
                kind=kind,
                processed=processed,
                use_cache=use_cache,
            )
            if f_ref is None:
                f_ref = f
            elif f.shape != f_ref.shape or not np.allclose(f, f_ref):
                raise ValueError("FRFs must share identical frequency grids.")
            H_rows.append(H)
        frf_matrix = np.vstack(H_rows)
        # Pass EVERYTHING to EMA.Model
        model = EMA.Model(
            frf_matrix,
            f_ref,
            **model_kwargs,
        )
        return model

    # ------------------------------------------------------------------ #
    # Multi-channel plotting
    # ------------------------------------------------------------------ #

    def _normalize_layout(self, layout: Any) -> list[list[Any]]:
        """
        Normalise a layout specification into a rectangular 2D list.

        Each cell in the returned 2D list is either:

        * ``None``,
        * a single :class:`Channel` or channel key, or
        * a sequence of :class:`Channel` or channel keys.

        Parameters
        ----------
        layout : Any
            Layout specification (sequence of rows or cells).

        Returns
        -------
        list of list
            Rectangular 2D list of layout cells.

        Raises
        ------
        TypeError
            If ``layout`` is not a sequence.
        ValueError
            If ``layout`` is empty or contains empty rows.
        """
        if not isinstance(layout, (list, tuple)):
            raise TypeError("layout must be a sequence of rows or cells.")
        if not layout:
            raise ValueError("layout must not be empty.")
        first = layout[0]
        if not isinstance(first, (list, tuple)):
            # 1D layout -> single row
            rows = [list(layout)]
        else:
            rows = [list(row) for row in layout]
        max_len = max(len(row) for row in rows)
        if max_len == 0:
            raise ValueError("layout rows must not be empty.")
        normalized: list[list[Any]] = []
        for row in rows:
            pad = max_len - len(row)
            if pad > 0:
                row = row + [None] * pad
            normalized.append(row)
        return normalized

    def _plot_one_channel(
        self,
        ch: Channel,
        ax: plt.Axes,
        plot_type: str,
        multi: bool,
        **kwargs: Any,
    ) -> None:
        """
        Internal helper to route ``plot_type`` to the appropriate channel method.

        For multi-channel axes, a generic quantity label is used on the
        y-axis and individual lines are distinguished by a legend. For
        single-channel axes, the channel's axis label is used instead.

        Parameters
        ----------
        ch : Channel
            Channel to plot.
        ax : matplotlib.axes.Axes
            Axes to plot on.
        plot_type : str
            Plot type specifier (time-history, Fourier, PSD, etc.).
        multi : bool
            If ``True``, multiple channels are plotted on the same axes.
        **kwargs
            Additional keyword arguments forwarded to the underlying
            plotting method.
        """
        # Decide label behaviour
        if multi:
            # Generic kind on the y-axis, individual lines distinguished by legend
            include_label = False
            include_kind = True
            include_legend = True
        else:
            include_label = True
            include_kind = False
            include_legend = False
        pt = plot_type.lower()
        if pt in ("time", "timehistory", "time_history"):
            ch.plot(
                ax=ax,
                include_label=include_label,
                include_kind=include_kind,
                include_legend=include_legend,
                **kwargs,
            )
        elif pt in ("fourier", "fft"):
            ch.plot_fourier(
                ax=ax,
                **kwargs,
            )
            # Fourier plot uses its own labels; legend optional for multi
            if multi:
                # Add a legend entry using the line label that Channel.plot would use
                line_label = ch.label_legend or ch.name_user or ch.name_input
                if line_label:
                    for line in ax.get_lines():
                        if line.get_label() == "_nolegend_":
                            line.set_label(line_label)
                            break
                    ax.legend()
        elif pt in ("psd", "welch", "power"):
            ch.plot_psd(
                ax=ax,
                **kwargs,
            )
            if multi:
                line_label = ch.label_legend or ch.name_user or ch.name_input
                if line_label:
                    for line in ax.get_lines():
                        if line.get_label() == "_nolegend_":
                            line.set_label(line_label)
                            break
                    ax.legend()
        elif pt in ("arias", "husid"):
            ch.plot_arias(
                ax=ax,
                **kwargs,
            )
        elif pt in ("response", "response_spectrum", "rs"):
            ch.plot_response_spectrum(
                ax=ax,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported plot_type {plot_type!r}. "
                "Expected 'timehistory', 'fourier', 'psd', 'arias', or 'response'."
            )

    def plot_grid(
        self,
        layout: Any | None = None,
        plot_type: str = "timehistory",
        sharex: bool = True,
        sharey: bool = True,
        title_suffix: str | None = None,
        make_caption: bool = False,
        **kwargs: Any,
    ):
        """
        Plot channels from this test in a grid of subplots.

        The ``layout`` argument describes how channels are arranged on the
        grid. Each cell in ``layout`` can be:

        * ``None`` : leave the subplot empty.
        * a single channel key / :class:`Channel` : one channel on that axes.
        * a sequence of channel keys / :class:`Channel` : multiple channels
          overlaid on the same axes, with a legend.

        If ``layout`` is ``None``, a single-row layout containing all
        channels (by index order) is used.

        Parameters
        ----------
        layout : Any, optional
            Layout specification (see above). If ``None``, all channels
            are arranged in a single row.
        plot_type : str, optional
            Plot type passed to the underlying channel-plotting methods
            (default ``"timehistory"``).
        sharex : bool, optional
            If ``True`` (default), share the x-axis among subplots.
        sharey : bool, optional
            If ``True`` (default), share the y-axis among subplots.
        title_suffix : str or None, optional
            Optional suffix appended to the figure title, after the test
            name (e.g. ``"Test 04: Accelerations"``).
        make_caption : bool, optional
            If ``True``, also return a simple figure caption describing
            the plotted channels.
        **kwargs
            Additional keyword arguments forwarded to the channel plotting
            methods (e.g. ``color``, ``fmax=...``, etc.).

        Returns
        -------
        fig, axes : (matplotlib.figure.Figure, np.ndarray of Axes)
            Figure and array of axes with the plotted channels.
        caption : str, optional
            If ``make_caption`` is ``True``, a third return value containing
            a simple caption string is provided.
        """
        # Default layout: single row with all channels
        if layout is None:
            layout = [list(range(len(self.channels)))]
        normalized = self._normalize_layout(layout)
        n_rows = len(normalized)
        n_cols = len(normalized[0])
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            layout="tight",
        )

        # Determine if any cell is a multiplot (contains >1 channel).
        # If so, we will render all subplots in "multi" mode (generic y-label + legend).
        global_multiplot = False
        for row in normalized:
            for cell in row:
                if isinstance(cell, (list, tuple)) and len(cell) > 1:
                    global_multiplot = True
                    break
            if global_multiplot:
                break

        all_channels: list[Channel] = []
        for i_row, row in enumerate(normalized):
            for j_col, cell in enumerate(row):
                ax = axes[i_row, j_col]
                if cell is None:
                    ax.set_visible(False)
                    continue
                if isinstance(cell, (list, tuple)):
                    keys = list(cell)
                else:
                    keys = [cell]
                channels: list[Channel] = []
                for key in keys:
                    if isinstance(key, Channel):
                        ch = key
                        # Use identity-based membership to avoid NumPy == issues
                        if not any(ch is existing for existing in self.channels):
                            raise ValueError(
                                "Channel in layout is not part of this Test."
                            )
                    else:
                        ch = self[key]
                    channels.append(ch)
                # If any subplot in the grid is a multiplot, render ALL subplots
                # in multi mode (generic axis label + legend). Otherwise use per-cell multi.
                if global_multiplot:
                    multi = True
                else:
                    multi = len(channels) > 1

                for ch in channels:
                    self._plot_one_channel(
                        ch=ch,
                        ax=ax,
                        plot_type=plot_type,
                        multi=multi,
                        **kwargs,
                    )
        full_title = (
            self.name if title_suffix is None else f"{self.name}: {title_suffix}"
        )
        fig.suptitle(full_title)
        if make_caption:
            # Build unique channel list preserving order
            uniq_channels: list[Channel] = []
            seen = set()
            for ch in all_channels:
                if id(ch) not in seen:
                    seen.add(id(ch))
                    uniq_channels.append(ch)
            channel_names = ", ".join(
                ch.name_user or ch.name_input or "<unnamed>" for ch in uniq_channels
            )
            caption = f"{full_title}. {plot_type} plots of channels: {channel_names}."
            return fig, axes, caption
        return fig, axes

    def plot_channels(
        self,
        selector: Any = None,
        ncols: int = 3,
        plot_type: str = "timehistory",
        sharex: bool = True,
        sharey: bool = True,
        title_suffix: str | None = None,
        make_caption: bool = False,
        **kwargs: Any,
    ):
        """
        Plot a list of channels from this test in a grid with fixed columns.

        Channels are selected via ``selector`` and arranged row-wise into
        subplots with ``ncols`` columns. This is convenient for plotting
        many similar channels (e.g. all accelerograms) at once.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to plot (index, name, :class:`Channel`, list, slice…).
            If ``None`` (default), all channels are plotted.
        ncols : int, optional
            Number of columns in the subplot grid (default is 3).
        plot_type : str, optional
            Plot type passed to the underlying channel-plotting methods
            (default ``"timehistory"``).
        sharex : bool, optional
            If ``True`` (default), share the x-axis among subplots.
        sharey : bool, optional
            If ``True`` (default), share the y-axis among subplots.
        title_suffix : str or None, optional
            Optional suffix appended to the figure title.
        make_caption : bool, optional
            If ``True``, also return a simple caption string.
        **kwargs
            Additional keyword arguments forwarded to the channel plotting
            methods.

        Returns
        -------
        fig, axes : (matplotlib.figure.Figure, np.ndarray of Axes)
            Figure and array of axes with the plotted channels.
        caption : str, optional
            If ``make_caption`` is ``True``, a third return value containing
            a simple caption string is provided.

        Raises
        ------
        ValueError
            If no channels are selected for plotting.
        """
        channels = list(self.iter_channels(selector))
        if not channels:
            raise ValueError("No channels selected for plotting.")
        rows: list[list[Channel]] = []
        for i in range(0, len(channels), ncols):
            rows.append(channels[i : i + ncols])
        return self.plot_grid(
            rows,
            plot_type=plot_type,
            sharex=sharex,
            sharey=sharey,
            title_suffix=title_suffix,
            make_caption=make_caption,
            **kwargs,
        )

    def channel_health(
        self,
        selector: ChannelSelector = None,
        fraction_for_event: float = 0.05,
        min_peak: float | None = None,
        min_snr: float = 3.0,
        min_crest: float = 2.0,
        processed: bool = True,
        use_cache: bool = True,
        tablefmt: str = "github",
    ) -> str:
        """
        Check basic health of selected channels using simple time-domain metrics.

        The method computes peak, RMS, crest factor, pre-event RMS and
        a simple SNR-like ratio for each channel, and returns a tabulated
        string summarising the results.

        Parameters
        ----------
        selector : ChannelSelector, optional
            Channels to inspect. If ``None``, all channels are checked.
        fraction_for_event : float, optional
            Fraction of the peak used to define the event window.
        min_peak : float or None, optional
            Minimum acceptable peak amplitude. If ``None``, this criterion
            is not applied.
        min_snr : float, optional
            Minimum RMS(event)/RMS(pre-event) ratio for a healthy response.
        min_crest : float, optional
            Minimum crest factor (peak/RMS) for a healthy response.
        processed : bool, optional
            Passed to :meth:`Channel.xy` (default ``True``).
        use_cache : bool, optional
            Passed to :meth:`Channel.xy` (default ``True``).
        tablefmt : str, optional
            Table format passed to :func:`tabulate` (default ``"github"``).

        Returns
        -------
        str
            Health check table as a formatted string. If no channels are
            selected, a short message is returned.

        Notes
        -----
        The ``status`` column is a simple qualitative label based on the
        peak amplitude, crest factor and SNR thresholds and should be
        interpreted as a quick screening rather than a rigorous test.
        """
        headers = [
            "idx",
            "name",
            "peak",
            "rms",
            "crest",
            "t_start",
            "t_end",
            "dur",
            "rms_pre",
            "snr_pre",
            "status",
        ]
        rows = []
        channels = list(self.iter_channels(selector=selector))
        if not channels:
            return "No channels selected for health check."
        for ch in channels:
            # Find index by identity, not by equality (avoids NumPy == issues)
            idx = None
            for i, existing in enumerate(self.channels):
                if ch is existing:
                    idx = i
                    break
            if idx is None:
                raise ValueError("Channel in selector is not part of this Test.")
            name = ch.name_user or ch.name_input or f"ch{idx}"
            try:
                t, y = ch.xy(processed=processed, use_cache=use_cache)
            except Exception as err:  # noqa: BLE001 - report any failure in the table
                rows.append(
                    [idx, name] + ["-"] * (len(headers) - 3) + [f"error: {err}"]
                )
                continue
            if y.size == 0:
                rows.append([idx, name, 0, 0, "-", "-", "-", "-", "-", "-", "empty"])
                continue
            peak = float(np.max(np.abs(y)))
            rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
            crest = peak / rms if rms > 0 else float("nan")
            t_start = t_end = dur = rms_pre = snr_pre = float("nan")
            if peak <= 0.0:
                status = "dead"
            else:
                thr = fraction_for_event * peak
                mask = np.abs(y) >= thr
                if not np.any(mask):
                    status = "no event"
                else:
                    i_start = int(np.argmax(mask))
                    i_end = int(len(mask) - 1 - np.argmax(mask[::-1]))
                    t_start = float(t[i_start])
                    t_end = float(t[i_end])
                    dur = t_end - t_start
                    if i_start > 0:
                        y_pre = y[:i_start]
                        rms_pre = float(np.sqrt(np.mean(y_pre**2)))
                    else:
                        rms_pre = float("nan")
                    y_evt = y[i_start : i_end + 1]
                    rms_evt = float(np.sqrt(np.mean(y_evt**2)))
                    snr_pre = (
                        rms_evt / rms_pre
                        if (rms_pre > 0 and np.isfinite(rms_pre))
                        else float("nan")
                    )
                    ok_peak = (min_peak is None) or (peak >= min_peak)
                    ok_snr = np.isfinite(snr_pre) and (snr_pre >= min_snr)
                    ok_crest = np.isfinite(crest) and (crest >= min_crest)
                    if ok_peak and ok_snr and ok_crest:
                        status = "ok"
                    elif ok_peak and (ok_snr or ok_crest):
                        status = "weak response"
                    else:
                        status = "noise"
            rows.append(
                [
                    idx,
                    name,
                    f"{peak:.3g}",
                    f"{rms:.3g}",
                    f"{crest:.3g}" if np.isfinite(crest) else "-",
                    f"{t_start:.3g}" if np.isfinite(t_start) else "-",
                    f"{t_end:.3g}" if np.isfinite(t_end) else "-",
                    f"{dur:.3g}" if np.isfinite(dur) else "-",
                    f"{rms_pre:.3g}" if np.isfinite(rms_pre) else "-",
                    f"{snr_pre:.3g}" if np.isfinite(snr_pre) else "-",
                    status,
                ]
            )
        return tabulate(
            rows,
            headers=headers,
            tablefmt=tablefmt,
            numalign="right",
            stralign="left",
        )
