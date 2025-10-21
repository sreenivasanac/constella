"""HTML templates used by visualization helpers."""

from __future__ import annotations


def build_umap_hover_html(
    *,
    data_script_path: str | None,
    title_json: str,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    """Return an HTML document embedding an interactive hoverable UMAP scatter."""

    script_include = ""
    if data_script_path:
        script_include = f"  <script src=\"{data_script_path}\" defer></script>\n"

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>UMAP Projection</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      background: #f9fafb;
      color: #111827;
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      font-size: 1.5rem;
      margin-bottom: 16px;
    }}
    .plot-container {{
      position: relative;
      max-width: {width}px;
    }}
    #umap-plot {{
      width: 100%;
      height: auto;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }}
    .tooltip {{
      position: absolute;
      display: none;
      min-width: 220px;
      max-width: 320px;
      padding: 12px;
      background: rgba(17, 24, 39, 0.92);
      color: #f9fafb;
      border-radius: 8px;
      pointer-events: none;
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.2);
      white-space: pre-wrap;
      line-height: 1.4;
      z-index: 10;
    }}
    .tooltip strong {{
      display: block;
      font-size: 0.95rem;
      margin-bottom: 4px;
    }}
    .tooltip .tooltip-id {{
      font-size: 0.85rem;
      opacity: 0.85;
      margin-bottom: 4px;
    }}
    .tooltip .tooltip-text {{
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <h1 id=\"plot-title\"></h1>
  <div class=\"plot-container\">
    <svg id=\"umap-plot\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-labelledby=\"plot-title\"></svg>
    <div id=\"tooltip\" class=\"tooltip\"></div>
  </div>
{script_include}  <script>
    const DATA_READY_EVENT = "umap-data-ready";
    const pageTitle = {title_json};
    const width = {width};
    const height = {height};
    const padding = 40;
    const xMin = {x_min};
    const xMax = {x_max};
    const yMin = {y_min};
    const yMax = {y_max};

    const svg = document.getElementById("umap-plot");
    const tooltip = document.getElementById("tooltip");

    svg.setAttribute("width", width);
    svg.setAttribute("height", height);

    let plotInitialized = false;

    function getExternalData() {{
      if (Array.isArray(window.UMAP_DATA)) {{
        return window.UMAP_DATA;
      }}
      return null;
    }}

    function scaleX(value) {{
      if (xMax - xMin === 0) {{
        return width / 2;
      }}
      return padding + ((value - xMin) / (xMax - xMin)) * (width - padding * 2);
    }}

    function scaleY(value) {{
      if (yMax - yMin === 0) {{
        return height / 2;
      }}
      return height - padding - ((value - yMin) / (yMax - yMin)) * (height - padding * 2);
    }}

    function showTooltip(evt, point) {{
      tooltip.innerHTML = "";

      const labelLine = document.createElement("strong");
      const labelValue = point && point.label !== undefined ? point.label : "Unknown";
      labelLine.textContent = "Cluster " + labelValue;
      tooltip.appendChild(labelLine);

      const idLine = document.createElement("div");
      idLine.className = "tooltip-id";
      idLine.textContent = point && point.identifier ? point.identifier : "";
      tooltip.appendChild(idLine);

      const textLine = document.createElement("div");
      textLine.className = "tooltip-text";
      textLine.textContent = point && point.text ? point.text : "";
      tooltip.appendChild(textLine);

      tooltip.style.display = "block";
      const pageX = evt && typeof evt.pageX === "number" ? evt.pageX : 0;
      const pageY = evt && typeof evt.pageY === "number" ? evt.pageY : 0;
      tooltip.style.left = String(pageX + 12) + "px";
      tooltip.style.top = String(pageY + 12) + "px";
    }}

    function moveTooltip(evt) {{
      const pageX = evt && typeof evt.pageX === "number" ? evt.pageX : 0;
      const pageY = evt && typeof evt.pageY === "number" ? evt.pageY : 0;
      tooltip.style.left = String(pageX + 12) + "px";
      tooltip.style.top = String(pageY + 12) + "px";
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function renderPlot(data) {{
      if (plotInitialized) {{
        return;
      }}
      if (!Array.isArray(data)) {{
        return;
      }}

      plotInitialized = true;
      document.getElementById("plot-title").textContent = pageTitle;
      svg.textContent = "";

      data.forEach((point) => {{
        const coordX = point && typeof point.x === "number" ? point.x : 0;
        const coordY = point && typeof point.y === "number" ? point.y : 0;
        const fillColor = point && point.color ? point.color : "#1f77b4";
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", scaleX(coordX));
        circle.setAttribute("cy", scaleY(coordY));
        circle.setAttribute("r", 5);
        circle.setAttribute("fill", fillColor);
        circle.setAttribute("stroke", "#1f2937");
        circle.setAttribute("stroke-width", 0.5);

        circle.addEventListener("mouseenter", (event) => showTooltip(event, point));
        circle.addEventListener("mousemove", moveTooltip);
        circle.addEventListener("mouseleave", hideTooltip);

        svg.appendChild(circle);
      }});
    }}

    function scheduleRender() {{
      const immediateData = getExternalData();
      if (Array.isArray(immediateData)) {{
        renderPlot(immediateData);
        return;
      }}

      document.addEventListener(
        DATA_READY_EVENT,
        () => {{
          const readyData = getExternalData();
          if (Array.isArray(readyData)) {{
            renderPlot(readyData);
          }}
        }},
        {{ once: true }}
      );
    }}

    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", scheduleRender, {{ once: true }});
    }} else {{
      scheduleRender();
    }}
  </script>
</body>
</html>
"""
