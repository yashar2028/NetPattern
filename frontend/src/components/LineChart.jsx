import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { LineChart as EChartsLine } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

import { formatMetric } from "../utils/format";

echarts.use([EChartsLine, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

const DARK_QUERY = "(prefers-color-scheme: dark)";

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Re-render when the color scheme flips, so the chart picks up the dark-mode tokens. */
function useColorScheme() {
  const [dark, setDark] = useState(() => window.matchMedia(DARK_QUERY).matches);
  useEffect(() => {
    const query = window.matchMedia(DARK_QUERY);
    const listener = (event) => setDark(event.matches);
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);
  return dark;
}

/**
 * Lines on one shared axis (never two y-scales). `series` is [{name, data: [[x, y], ...]}];
 * colors follow the series position in fixed order. A legend appears for two or more series.
 */
export default function LineChart({ series, xName, label, height = 260 }) {
  const element = useRef(null);
  const chart = useRef(null);
  const dark = useColorScheme();

  useEffect(() => {
    chart.current = echarts.init(element.current);
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.current.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    const muted = token("--text-muted");
    const grid = token("--border");
    const text = token("--text");
    const surface = token("--surface");
    const legend = series.length > 1;
    chart.current.setOption(
      {
        animation: false,
        textStyle: { fontFamily: "inherit", color: muted },
        grid: { left: 56, right: 20, top: legend ? 36 : 16, bottom: 40 },
        legend: legend ? { top: 0, left: 0, icon: "roundRect", textStyle: { color: text } } : { show: false },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "line", lineStyle: { color: muted } },
          backgroundColor: surface,
          borderColor: grid,
          textStyle: { color: text },
          valueFormatter: (value) => formatMetric(value),
        },
        xAxis: {
          type: "value",
          name: xName,
          nameLocation: "middle",
          nameGap: 26,
          minInterval: 1,
          min: "dataMin",
          max: "dataMax",
          axisLine: { lineStyle: { color: grid } },
          axisLabel: { color: muted },
          splitLine: { show: false },
        },
        yAxis: {
          type: "value",
          scale: true,
          axisLabel: { color: muted },
          splitLine: { lineStyle: { color: grid } },
        },
        series: series.map((item, index) => {
          const color = token(`--series-${index + 1}`);
          return {
            name: item.name,
            type: "line",
            data: item.data,
            color,
            lineStyle: { width: 2 },
            symbol: "circle",
            symbolSize: item.data.length > 30 ? 0 : 8,
            itemStyle: { color, borderColor: surface, borderWidth: 2 },
          };
        }),
      },
      true
    );
  }, [series, xName, dark]);

  return <div ref={element} className="chart" style={{ height }} role="img" aria-label={label} />;
}
