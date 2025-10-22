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
    .axes line,
    .ticks line {{
      stroke: #4b5563;
      stroke-width: 1;
      pointer-events: none;
    }}
    .ticks text {{
      fill: #1f2937;
      font-size: 0.7rem;
    }}
    .axis-label {{
      fill: #1f2937;
      font-size: 0.8rem;
      font-weight: 600;
      pointer-events: none;
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
    .legend {{
      display: none;
      position: absolute;
      bottom: 20px;
      left: 20px;
      max-width: 240px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(17, 24, 39, 0.15);
      border-radius: 6px;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
      gap: 8px;
      grid-template-columns: 1fr;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: #111827;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(17, 24, 39, 0.4);
      flex-shrink: 0;
    }}
  </style>
</head>
<body>
  <h1 id=\"plot-title\"></h1>
  <div class=\"plot-container\">
    <svg id=\"umap-plot\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-labelledby=\"plot-title\"></svg>
    <div id=\"tooltip\" class=\"tooltip\"></div>
    <div id="legend" class="legend" aria-label="Cluster color mapping"></div>
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
    const legendContainer = document.getElementById("legend");

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

    function generateTicks(min, max, steps = 5) {{
      if (!Number.isFinite(min) || !Number.isFinite(max) || steps <= 0) {{
        return [];
      }}
      if (min === max) {{
        return [min];
      }}
      const increment = (max - min) / steps;
      const ticks = [];
      for (let i = 0; i <= steps; i += 1) {{
        ticks.push(min + increment * i);
      }}
      return ticks;
    }}

    function createSvgLine(x1, y1, x2, y2, className) {{
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      if (className) {{
        line.setAttribute("class", className);
      }}
      return line;
    }}

    function createSvgText(x, y, textContent, className) {{
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y);
      text.textContent = textContent;
      if (className) {{
        text.setAttribute("class", className);
      }}
      return text;
    }}

    function formatTick(value) {{
      if (!Number.isFinite(value)) {{
        return "";
      }}
      const fixed = value.toFixed(2);
      return fixed === "-0.00" ? "0.00" : fixed;
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

      const pointsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      pointsGroup.setAttribute("class", "points");
      const axesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      axesGroup.setAttribute("class", "axes");
      const ticksGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      ticksGroup.setAttribute("class", "ticks");
      const labelsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      labelsGroup.setAttribute("class", "axis-labels");

      svg.appendChild(pointsGroup);
      svg.appendChild(axesGroup);
      svg.appendChild(ticksGroup);
      svg.appendChild(labelsGroup);

      const axes = [
        {{ x1: padding, y1: height - padding, x2: width - padding, y2: height - padding }},
        {{ x1: padding, y1: padding, x2: width - padding, y2: padding }},
        {{ x1: padding, y1: padding, x2: padding, y2: height - padding }},
        {{ x1: width - padding, y1: padding, x2: width - padding, y2: height - padding }},
      ];

      axes.forEach((coords) => {{
        axesGroup.appendChild(createSvgLine(coords.x1, coords.y1, coords.x2, coords.y2));
      }});

      const xTicks = generateTicks(xMin, xMax, 5);
      const yTicks = generateTicks(yMin, yMax, 5);
      const tickLength = 8;

      xTicks.forEach((tickValue) => {{
        const xPos = scaleX(tickValue);
        ticksGroup.appendChild(createSvgLine(xPos, height - padding, xPos, height - padding + tickLength));
        ticksGroup.appendChild(createSvgLine(xPos, padding, xPos, padding - tickLength));

        const labelText = formatTick(tickValue);
        const bottomLabel = createSvgText(xPos, height - padding + tickLength + 10, labelText, "tick-label");
        bottomLabel.setAttribute("text-anchor", "middle");
        bottomLabel.setAttribute("dominant-baseline", "hanging");
        ticksGroup.appendChild(bottomLabel);

        const topLabel = createSvgText(xPos, padding - tickLength - 4, labelText, "tick-label");
        topLabel.setAttribute("text-anchor", "middle");
        topLabel.setAttribute("dominant-baseline", "baseline");
        ticksGroup.appendChild(topLabel);
      }});

      yTicks.forEach((tickValue) => {{
        const yPos = scaleY(tickValue);
        ticksGroup.appendChild(createSvgLine(padding, yPos, padding - tickLength, yPos));
        ticksGroup.appendChild(createSvgLine(width - padding, yPos, width - padding + tickLength, yPos));

        const labelText = formatTick(tickValue);
        const leftLabel = createSvgText(padding - tickLength - 6, yPos, labelText, "tick-label");
        leftLabel.setAttribute("text-anchor", "end");
        leftLabel.setAttribute("dominant-baseline", "middle");
        ticksGroup.appendChild(leftLabel);

        const rightLabel = createSvgText(width - padding + tickLength + 6, yPos, labelText, "tick-label");
        rightLabel.setAttribute("text-anchor", "start");
        rightLabel.setAttribute("dominant-baseline", "middle");
        ticksGroup.appendChild(rightLabel);
      }});

      const xAxisLabel = createSvgText(width / 2, height - padding + 36, "UMAP 1", "axis-label");
      xAxisLabel.setAttribute("text-anchor", "middle");
      xAxisLabel.setAttribute("dominant-baseline", "hanging");
      labelsGroup.appendChild(xAxisLabel);

      const yAxisLabel = createSvgText(padding - 44, height / 2, "UMAP 2", "axis-label");
      yAxisLabel.setAttribute("text-anchor", "middle");
      yAxisLabel.setAttribute("dominant-baseline", "middle");
      yAxisLabel.setAttribute(
        "transform",
        "rotate(-90 " + (padding - 44) + " " + height / 2 + ")"
      );
      labelsGroup.appendChild(yAxisLabel);

      const legendItems = new Map();

      data.forEach((point) => {{
        const coordX = point && typeof point.x === "number" ? point.x : 0;
        const coordY = point && typeof point.y === "number" ? point.y : 0;
        const fillColor = point && point.color ? point.color : "#1f77b4";
        const clusterLabel = point && point.label !== undefined ? String(point.label) : "Unknown";

        if (!legendItems.has(clusterLabel)) {{
          legendItems.set(clusterLabel, fillColor);
        }}

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

        pointsGroup.appendChild(circle);
      }});

      if (legendContainer) {{
        legendContainer.textContent = "";
        if (legendItems.size > 0) {{
          legendContainer.style.display = "grid";
          legendItems.forEach((color, label) => {{
            const item = document.createElement("div");
            item.setAttribute("class", "legend-item");

            const swatch = document.createElement("span");
            swatch.setAttribute("class", "legend-swatch");
            swatch.style.backgroundColor = color;

            const text = document.createElement("span");
            text.textContent = label;

            item.appendChild(swatch);
            item.appendChild(text);
            legendContainer.appendChild(item);
          }});
        }} else {{
          legendContainer.style.display = "none";
        }}
      }}
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
