"""HTML templates used by visualization helpers."""

from __future__ import annotations


def build_umap_hover_html(
    *,
    data_json: str,
    title_json: str,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    """Return an HTML document embedding an interactive hoverable UMAP scatter."""

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
  <script>
    const data = {data_json};
    const pageTitle = {title_json};
    const width = {width};
    const height = {height};
    const padding = 40;
    const xMin = {x_min};
    const xMax = {x_max};
    const yMin = {y_min};
    const yMax = {y_max};

    document.getElementById("plot-title").textContent = pageTitle;

    const svg = document.getElementById("umap-plot");
    const tooltip = document.getElementById("tooltip");

    svg.setAttribute("width", width);
    svg.setAttribute("height", height);

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

    function showTooltip(event, point) {{
      tooltip.innerHTML = "";

      const labelLine = document.createElement("strong");
      labelLine.textContent = `Cluster ${'${'}point.label${'}'}`;
      tooltip.appendChild(labelLine);

      const idLine = document.createElement("div");
      idLine.className = "tooltip-id";
      idLine.textContent = point.identifier;
      tooltip.appendChild(idLine);

      const textLine = document.createElement("div");
      textLine.className = "tooltip-text";
      textLine.textContent = point.text;
      tooltip.appendChild(textLine);

      tooltip.style.display = "block";
      tooltip.style.left = `${'${'}event.pageX + 12${'}'}px`;
      tooltip.style.top = `${'${'}event.pageY + 12${'}'}px`;
    }}

    function moveTooltip(event) {{
      tooltip.style.left = `${'${'}event.pageX + 12${'}'}px`;
      tooltip.style.top = `${'${'}event.pageY + 12${'}'}px`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    data.forEach((point) => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", scaleX(point.x));
      circle.setAttribute("cy", scaleY(point.y));
      circle.setAttribute("r", 5);
      circle.setAttribute("fill", point.color);
      circle.setAttribute("stroke", "#1f2937");
      circle.setAttribute("stroke-width", 0.5);

      circle.addEventListener("mouseenter", (event) => showTooltip(event, point));
      circle.addEventListener("mousemove", moveTooltip);
      circle.addEventListener("mouseleave", hideTooltip);

      svg.appendChild(circle);
    }});
  </script>
</body>
</html>
"""
