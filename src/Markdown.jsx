import { useState } from "react";

const mono = `'SF Mono','Menlo','Consolas',monospace`;
const C = {
  bg: "#0b0b0f", s1: "#131318", s2: "#1b1b22", s3: "#232330",
  border: "#282838", text: "#d4d4e4", dim: "#65657e", accent: "#6e56cf",
  green: "#30a46c",
};

const safeHref = (url) => /^(https?:|mailto:)/i.test(url.trim()) ? url.trim() : null;

// ── Inline parser: bold, italic, strikethrough, inline code, links ──
function parseInline(text, keyBase = "i") {
  const nodes = [];
  const re = /(`[^`]+`)|(\*\*.+?\*\*|__.+?__)|(~~.+?~~)|(\*(?!\s).+?(?<!\s)\*|_(?!\s).+?(?<!\s)_)|(\[[^\]]*\]\([^)\s]+\))/;
  let rest = text;
  let k = 0;
  while (rest.length) {
    const m = rest.match(re);
    if (!m) { nodes.push(rest); break; }
    if (m.index > 0) nodes.push(rest.slice(0, m.index));
    const tok = m[0];
    const key = `${keyBase}-${k++}`;
    if (m[1]) {
      nodes.push(<code key={key} style={{ fontFamily: mono, fontSize: "0.88em", background: C.s3, padding: "1px 5px", borderRadius: 4, color: C.green }}>{tok.slice(1, -1)}</code>);
    } else if (m[2]) {
      nodes.push(<strong key={key} style={{ fontWeight: 700 }}>{parseInline(tok.slice(2, -2), key)}</strong>);
    } else if (m[3]) {
      nodes.push(<s key={key} style={{ opacity: 0.6 }}>{parseInline(tok.slice(2, -2), key)}</s>);
    } else if (m[4]) {
      nodes.push(<em key={key} style={{ fontStyle: "italic" }}>{parseInline(tok.slice(1, -1), key)}</em>);
    } else if (m[5]) {
      const lm = tok.match(/^\[([^\]]*)\]\(([^)\s]+)\)$/);
      const href = safeHref(lm[2]);
      nodes.push(href
        ? <a key={key} href={href} target="_blank" rel="noopener noreferrer" style={{ color: C.accent, textDecoration: "underline" }}>{parseInline(lm[1], key)}</a>
        : <span key={key}>{lm[1]}</span>);
    }
    rest = rest.slice(m.index + tok.length);
  }
  return nodes;
}

function CodeBlock({ code, lang }) {
  const [copied, setCopied] = useState(false);
  const onCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div style={{ position: "relative", margin: "10px 0" }}>
      <button onClick={onCopy} style={{
        position: "absolute", top: 6, right: 6, padding: "3px 9px", fontSize: 10, fontFamily: mono,
        fontWeight: 600, borderRadius: 5, border: "none", cursor: "pointer",
        background: copied ? C.green : C.s3, color: copied ? "#000" : C.dim,
      }}>{copied ? "✓" : "copy"}</button>
      {lang && <div style={{ fontSize: 10, fontFamily: mono, color: C.dim, padding: "4px 12px", background: C.s3, borderRadius: "6px 6px 0 0" }}>{lang}</div>}
      <pre style={{
        margin: 0, padding: "12px 14px", background: C.bg, border: `1px solid ${C.border}`,
        borderRadius: lang ? "0 0 6px 6px" : 6, overflowX: "auto",
      }}>
        <code style={{ fontFamily: mono, fontSize: 12, lineHeight: 1.5, color: C.text, whiteSpace: "pre" }}>{code}</code>
      </pre>
    </div>
  );
}

// ── Block parser ──
export default function Markdown({ text, style }) {
  const lines = String(text).replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    const fence = line.match(/^\s*```(.*)$/);
    if (fence) {
      const lang = fence[1].trim();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // closing fence
      blocks.push({ type: "code", code: buf.join("\n"), lang });
      continue;
    }

    // Blank line
    if (!line.trim()) { i++; continue; }

    // Heading
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) { blocks.push({ type: "heading", level: h[1].length, text: h[2] }); i++; continue; }

    // Horizontal rule
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) { blocks.push({ type: "hr" }); i++; continue; }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
      blocks.push({ type: "quote", text: buf.join("\n") });
      continue;
    }

    // List (ordered or unordered)
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const items = [];
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
        i++;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }

    // Table (GitHub-flavored)
    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const splitRow = (r) => r.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(c => c.trim());
      const header = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // Paragraph — consume consecutive non-blank, non-special lines
    const buf = [];
    while (
      i < lines.length && lines[i].trim() &&
      !/^\s*```/.test(lines[i]) &&
      !/^(#{1,6})\s+/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i]) &&
      !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])
    ) {
      buf.push(lines[i]);
      i++;
    }
    blocks.push({ type: "para", text: buf.join("\n") });
  }

  const hSizes = { 1: 22, 2: 18, 3: 16, 4: 14, 5: 13, 6: 12 };

  return (
    <div style={{ fontSize: 13, lineHeight: 1.6, color: C.text, ...style }}>
      {blocks.map((b, idx) => {
        if (b.type === "code") return <CodeBlock key={idx} code={b.code} lang={b.lang} />;
        if (b.type === "hr") return <hr key={idx} style={{ border: "none", borderTop: `1px solid ${C.border}`, margin: "14px 0" }} />;
        if (b.type === "heading") {
          const Tag = `h${b.level}`;
          return <Tag key={idx} style={{ fontSize: hSizes[b.level], fontWeight: 700, margin: "14px 0 6px", lineHeight: 1.3 }}>{parseInline(b.text, `h${idx}`)}</Tag>;
        }
        if (b.type === "quote") {
          return (
            <blockquote key={idx} style={{ margin: "8px 0", padding: "4px 14px", borderLeft: `3px solid ${C.accent}`, color: C.dim }}>
              {b.text.split("\n").map((l, j) => <div key={j}>{parseInline(l, `q${idx}-${j}`)}</div>)}
            </blockquote>
          );
        }
        if (b.type === "list") {
          const Tag = b.ordered ? "ol" : "ul";
          return (
            <Tag key={idx} style={{ margin: "8px 0", paddingLeft: 22 }}>
              {b.items.map((it, j) => <li key={j} style={{ margin: "3px 0" }}>{parseInline(it, `l${idx}-${j}`)}</li>)}
            </Tag>
          );
        }
        if (b.type === "table") {
          return (
            <div key={idx} style={{ overflowX: "auto", margin: "10px 0" }}>
              <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
                <thead>
                  <tr>{b.header.map((c, j) => (
                    <th key={j} style={{ border: `1px solid ${C.border}`, padding: "6px 10px", textAlign: "left", background: C.s2, fontWeight: 700 }}>{parseInline(c, `th${idx}-${j}`)}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {b.rows.map((row, j) => (
                    <tr key={j}>{b.header.map((_, c) => (
                      <td key={c} style={{ border: `1px solid ${C.border}`, padding: "6px 10px" }}>{parseInline(row[c] || "", `td${idx}-${j}-${c}`)}</td>
                    ))}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        // paragraph
        return (
          <p key={idx} style={{ margin: "8px 0", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {b.text.split("\n").map((l, j) => (
              <span key={j}>{j > 0 && <br />}{parseInline(l, `p${idx}-${j}`)}</span>
            ))}
          </p>
        );
      })}
    </div>
  );
}
