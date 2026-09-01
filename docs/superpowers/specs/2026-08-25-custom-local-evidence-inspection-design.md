# Custom Local Evidence Inspection Design

Add one output-free notebook cell for inspecting any zero-based text index in
one HG or MG split. It will take `CUSTOM_TEXT_INDEX`, `CUSTOM_SPLIT`, and
`CUSTOM_WINDOW_LENGTH`; resolve the persistent text ID; select that text's
cached scored windows; render the local-evidence timeline; and render the
selected strongest token range with the existing heatmap. It will raise a
clear error when the requested text/window combination is unavailable.
