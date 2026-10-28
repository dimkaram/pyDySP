# Design: results and plots (Stages 2 and 3)

One result class serves both scalar results (Stage 2) and curve results
(Stage 3). The x-axis, selection and export logic are shared, and only the plot
method differs.

## Scalar results: `metric` (Stage 2)

`campaign.metric(channel, what)` reduces one channel in each test to one number
and returns a result object.

```python
a_max = camp.metric("A1", "max_abs")      # built-in, by name
l1    = camp.metric("L1", residual_fn)    # optional: any function of a Channel
```

- **Built-in names:** `"max"`, `"min"`, `"max_abs"`, `"rms"`, and the mean of the
  last N samples (the residual displacement used in the notebooks). They return
  only the value, not the `(time, value)` tuple that `Channel.max_abs()`
  returns.
- **Custom function:** the second argument may instead be a function that
  receives a `Channel` and returns a number. The built-in names are ready-made
  functions of this kind. Documentation should focus on the names.
- **One method only.** There is no separate `map_channel`.
- **Missing channel:** NaN plus a warning, or an error with `strict=True`.

**What a result holds:** the values (one per test), the test numbers (default
x-axis), the test names, and a label and units taken from the channel.

**Transforms:** `.abs()`, negation (`-f_max`), `.cumsum()`, and arithmetic with
numbers or other results (`f_max / d_max`). For example, the accumulated
displacement of an LVDT is `camp.metric("L1", residual_fn).abs().cumsum()`.

**Export:** `.to_array()`, `.to_table()` (printable, via `tabulate`) and
`.to_csv(filename)`.

## Plotting conventions (Stage 2)

Plots return `(fig, ax)` and accept `ax=`, as `Test.plot_grid` and
`Test.plot_transfer_function` do, so users can build multi-panel figures and
apply their own style.

```python
fig, ax = a_max.plot()                  # value against test number
a_max.plot(ax=ax[0, 1])                 # draw into the user's own subplot
a_max.plot(highlight=21)                # mark one test (by number) with a square marker
f_max.plot(x=d_max)                     # one result against another
f_max.plot(x=d_max, origin=True)        # start the line at (0, 0)
```

- **No styling inside the class:** no fixed figure sizes, fonts, colours or
  captions. The user's `figure_style.mplstyle` applies, and users still set
  `fig.caption` and `fig.set_size_inches` themselves.
- **Axis labels** default to the channel name and units, and can be overridden
  (`ylabel=`).
- **`highlight=`** takes a test number and replaces the two-line square-marker
  overlay repeated in the notebooks.
- **Gaps:** a missing test number leaves a visible gap in the line, and NaN
  values are not drawn.
- **x-axis** defaults to test number. `plot(x="amp")` uses a `meta` key instead.

## Curve results: transfer functions and spectra (Stage 3)

Curve results (one curve per test) share the result class used for scalars.

```python
tf = camp.transfer_function("A8", "A14", kind="H1", nperseg=1024)
tf.plot()                                    # overlay: one line per test, labelled "Test 3", ...
tf.plot(kind="heatmap")                      # frequency on one axis, test number on the other
f_peak = tf.peak_frequency(fmin=5, fmax=40)  # a normal scalar result

sp = camp.spectrum("A1", method="welch")     # same container, same plots
```

- **Selection first:** narrow the campaign with `select` (for example only the
  white-noise tests), or pass `tests=[3, 29]` for a quick comparison of two
  tests.
- **Overlay** is the default plot. It replaces the loop with hand-built
  "Test n" labels, and it labels lines with real test numbers.
- **Heatmap** is an option. Frequency runs along one axis, test number along the
  other, and colour shows amplitude, so each test is one stripe. A resonance
  appears as a band that slides sideways as the structure softens. It is best on
  tests with the same input (white noise), because then any change comes from
  the system. Document it clearly as an extra; the author wants to try it on
  real data.
- **`peak_frequency`** turns a curve result into a scalar result, so a natural
  frequency can be plotted against test number like any other metric.
- **Frequency grid:** the heatmap requires all selected tests to share one
  frequency grid. If they differ, raise a clear error rather than interpolating
  silently.
- **Export:** `.to_csv()` writes frequency down the rows and one column per test.
- **Not in this version:** response spectra across tests.
