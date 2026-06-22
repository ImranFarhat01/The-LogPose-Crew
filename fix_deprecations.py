"""Fix all deprecation warnings in app.py"""

with open(r"D:\Flipkart GridLock 2.0\Round-2\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
changes = 0
for line in lines:
    original = line
    # Fix scatter_mapbox -> scatter_map
    line = line.replace("scatter_mapbox", "scatter_map")
    # Fix mapbox_style -> map_style
    line = line.replace("mapbox_style=", "map_style=")
    # Fix use_container_width for plotly_chart
    if "st.plotly_chart" in line and "use_container_width=True" in line:
        line = line.replace("use_container_width=True", "width='stretch'")
    # Fix use_container_width for st.image
    if "st.image" in line and "use_container_width=True" in line:
        line = line.replace("use_container_width=True", "width='stretch'")
    # Fix use_container_width for st.dataframe
    if "st.dataframe" in line and "use_container_width=True" in line:
        line = line.replace("use_container_width=True", "width='stretch'")
    if line != original:
        changes += 1
    new_lines.append(line)

with open(r"D:\Flipkart GridLock 2.0\Round-2\app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Fixed {changes} lines with deprecation warnings.")
