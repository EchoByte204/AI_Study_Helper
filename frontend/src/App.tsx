import { FormEvent, ReactNode, useEffect, useRef, useState } from "react";
import { BookOpen, Check, FileText, GraduationCap, LoaderCircle, MessageCircle, Plus, Send, Sparkles, Trash2, Upload, X } from "lucide-react";

type Doc = { id: string; filename: string; page_count: number; chunk_count: number; figure_count: number };
type Citation = { number: number; document_id: string; filename: string; page: number; excerpt: string; score: number; kind?: "text" | "figure"; image_url?: string };
type Figure = { number: number; id: string; document_id: string; filename: string; page_number: number; caption: string; image_url: string; score: number };
type AnswerClaim = { text: string; source_numbers: number[] };
type AnswerSection = { title: string; content: AnswerClaim[] };
type StructuredAnswer = { direct_answer: AnswerClaim; sections?: AnswerSection[]; explanation: AnswerClaim[]; formula?: AnswerClaim | null; formula_terms?: AnswerClaim[]; additional_context?: string[]; key_points: AnswerClaim[]; limitations: string[]; abstained: boolean };
type StudyGuide = { document_id: string; filename: string; slide_count: number; slides_processed: number; coverage_complete: boolean; empty_slides: number[]; sections: AnswerSection[]; figures: Figure[] };
type BackgroundTask = { task_id: string; task_type: "upload" | "study_guide"; status: "queued" | "running" | "completed" | "failed"; progress: number; message: string; document?: Doc; study_guide?: StudyGuide; error?: string };
type Turn = { role: "user" | "assistant"; text: string; citations?: Citation[]; figures?: Figure[]; structured?: StructuredAnswer; visualRequested?: boolean };
type ChatSession = { session_id: string; title: string; updated_at: string };
const api = "/api";

async function apiJson<T>(response: Response, operation: string): Promise<T> {
  const raw = await response.text();
  let body: any;
  try {
    body = raw ? JSON.parse(raw) : {};
  } catch {
    if (!response.ok) throw new Error(raw.trim().slice(0, 240) || `${operation} failed (${response.status})`);
    throw new Error(`${operation} returned an unexpected response. Check that the API server is running.`);
  }
  if (!response.ok) throw new Error(body.detail || `${operation} failed (${response.status})`);
  return body as T;
}

function uploadWithProgress(data: FormData, onProgress: (percent: number) => void): Promise<BackgroundTask> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${api}/documents`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 10));
    };
    request.onerror = () => reject(new Error("Upload connection failed. Check that the API is running."));
    request.onload = () => {
      let body: any;
      try { body = request.responseText ? JSON.parse(request.responseText) : {}; }
      catch { reject(new Error("Upload returned an unreadable response.")); return; }
      if (request.status < 200 || request.status >= 300) { reject(new Error(body.detail || `Upload failed (${request.status})`)); return; }
      resolve(body as BackgroundTask);
    };
    request.send(data);
  });
}

function inlineFormatting(text: string): ReactNode[] {
  const pieces = text.split(/(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*]+\*|_[^_]+_)/g).filter(Boolean);
  return pieces.map((piece, index) => {
    const displayMath = piece.startsWith("\\[") || piece.startsWith("$$");
    const inlineMath = piece.startsWith("\\(") || (piece.startsWith("$") && !displayMath);
    if (displayMath || inlineMath) return <MathFormula key={index} source={piece.replace(/^\\\[|\\\]$|^\\\(|\\\)$|^\$\$|\$\$$|^\$|\$$/g, "").trim()} display={displayMath} />;
    if ((piece.startsWith("**") && piece.endsWith("**")) || (piece.startsWith("__") && piece.endsWith("__"))) return piece.slice(2, -2);
    if (piece.startsWith("`")) return piece.slice(1, -1);
    if ((piece.startsWith("*") && piece.endsWith("*")) || (piece.startsWith("_") && piece.endsWith("_"))) return piece.slice(1, -1);
    return piece;
  });
}

function readLatexGroup(source: string, start: number): [string, number] {
  if (source[start] !== "{") return [source[start] || "", Math.min(start + 1, source.length)];
  let depth = 1;
  let index = start + 1;
  for (; index < source.length && depth > 0; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") depth -= 1;
  }
  return [source.slice(start + 1, depth === 0 ? index - 1 : index), index];
}

const mathSymbols: Record<string, string> = {
  alpha: "\u03b1", beta: "\u03b2", gamma: "\u03b3", delta: "\u03b4", theta: "\u03b8", lambda: "\u03bb", mu: "\u03bc", pi: "\u03c0", sigma: "\u03c3", phi: "\u03c6", omega: "\u03c9",
  times: "\u00d7", cdot: "\u00b7", pm: "\u00b1", le: "\u2264", leq: "\u2264", ge: "\u2265", geq: "\u2265", neq: "\u2260", approx: "\u2248", infty: "\u221e", percent: "%",
};

function latexNodes(source: string, keyPrefix = "m"): ReactNode[] {
  const nodes: ReactNode[] = [];
  let text = "";
  const flush = () => { if (text) { nodes.push(text); text = ""; } };
  for (let index = 0; index < source.length;) {
    if (source[index] === "\\") {
      const commandMatch = source.slice(index + 1).match(/^[a-zA-Z]+/);
      if (!commandMatch) { text += source[index + 1] || ""; index += 2; continue; }
      const command = commandMatch[0];
      index += command.length + 1;
      if (command === "frac") {
        flush();
        const [numerator, afterNumerator] = readLatexGroup(source, index);
        const [denominator, afterDenominator] = readLatexGroup(source, afterNumerator);
        nodes.push(<span className="math-frac" key={`${keyPrefix}-${index}`}><span>{latexNodes(numerator, `${keyPrefix}-n${index}`)}</span><span>{latexNodes(denominator, `${keyPrefix}-d${index}`)}</span></span>);
        index = afterDenominator;
      } else if (command === "text") {
        flush();
        const [value, nextIndex] = readLatexGroup(source, index);
        nodes.push(<span key={`${keyPrefix}-${index}`}>{value}</span>);
        index = nextIndex;
      } else if (command === "left" || command === "right") {
        // Delimiter sizing has no visual meaning in this lightweight math renderer.
      } else text += mathSymbols[command] || `\\${command}`;
      continue;
    }
    if (source[index] === "^" || source[index] === "_") {
      flush();
      const marker = source[index];
      const [value, nextIndex] = source[index + 1] === "{" ? readLatexGroup(source, index + 1) : [source[index + 1] || "", Math.min(index + 2, source.length)];
      const Tag = marker === "^" ? "sup" : "sub";
      nodes.push(<Tag key={`${keyPrefix}-${index}`}>{latexNodes(value, `${keyPrefix}-p${index}`)}</Tag>);
      index = nextIndex;
      continue;
    }
    if (source[index] === "{" || source[index] === "}") { index += 1; continue; }
    text += source[index];
    index += 1;
  }
  flush();
  return nodes;
}

function MathFormula({ source, display }: { source: string; display: boolean }) {
  return display ? <div className="math-display" role="math">{latexNodes(source)}</div> : <span className="math-inline" role="math">{latexNodes(source)}</span>;
}

function FormattedText({ text }: { text: string }) {
  const blocks: { type: "paragraph" | "ordered" | "unordered" | "table"; lines: string[] }[] = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("|") && line.endsWith("|")) {
      const last = blocks[blocks.length - 1];
      if (last?.type === "table") last.lines.push(line);
      else blocks.push({ type: "table", lines: [line] });
      continue;
    }
    const type = /^\d+[.)]\s+/.test(line) ? "ordered" : /^[-*\u2022]\s+/.test(line) ? "unordered" : "paragraph";
    const content = type === "ordered" ? line.replace(/^\d+[.)]\s+/, "") : type === "unordered" ? line.replace(/^[-*\u2022]\s+/, "") : line;
    const last = blocks[blocks.length - 1];
    if (last?.type === type && type !== "paragraph") last.lines.push(content);
    else blocks.push({ type, lines: [content] });
  }
  return <div className="formatted-text">{blocks.map((block, index) => {
    if (block.type === "table") {
      const rows = block.lines.map((line) => line.slice(1, -1).split("|").map((cell) => cell.trim()));
      const divider = rows[1]?.every((cell) => /^:?-{3,}:?$/.test(cell));
      const headers = divider ? rows[0] : undefined;
      const body = rows.slice(divider ? 2 : 0);
      return <div className="answer-table-wrap" key={index}><table><tbody>{headers && <tr>{headers.map((cell, cellIndex) => <th key={cellIndex}>{inlineFormatting(cell)}</th>)}</tr>}{body.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{inlineFormatting(cell)}</td>)}</tr>)}</tbody></table></div>;
    }
    if (block.type === "ordered") return <ol key={index}>{block.lines.map((line, itemIndex) => <li key={itemIndex}>{inlineFormatting(line)}</li>)}</ol>;
    if (block.type === "unordered") return <ul key={index}>{block.lines.map((line, itemIndex) => <li key={itemIndex}>{inlineFormatting(line)}</li>)}</ul>;
    return <div className="formatted-paragraph" key={index}>{inlineFormatting(block.lines[0])}</div>;
  })}</div>;
}

function formulaMarkup(source: string) {
  return /\\\[|\\\(|\$\$?/.test(source) ? source : `\\[${source}\\]`;
}

function isFigureCited(figure: Figure, answer?: StructuredAnswer) {
  if (!answer || answer.abstained) return false;
  const claims = [answer.direct_answer, ...(answer.sections || []).flatMap((section) => section.content), ...answer.explanation, ...answer.key_points];
  return claims.some((claim) => claim.source_numbers.includes(figure.number));
}

function StudyNoteSection({ section, index, figures }: { section: AnswerSection; index: number; figures: Figure[] }) {
  const slides = new Set(section.content.flatMap((claim) => claim.source_numbers));
  const sectionFigures = figures.filter((figure) => slides.has(figure.page_number)).slice(0, 3);
  return <details className="study-note-section" id={`study-section-${index}`} open><summary><span>{index + 1}. {section.title}</span><span className="study-section-toggle" aria-hidden="true">⌄</span></summary><div className="study-section-body">{section.content.map((claim, claimIndex) => <div className="answer-detail-paragraph" key={claimIndex}><FormattedText text={claim.text} /></div>)}{!!sectionFigures.length && <div className="study-section-figures">{sectionFigures.map((figure) => <figure key={figure.id}><img src={figure.image_url} alt={`Visual on slide ${figure.page_number} of ${figure.filename}`} loading="lazy" /><figcaption>Slide {figure.page_number} · {figure.filename}</figcaption></figure>)}</div>}</div></details>;
}

function StudyGuidePage({ guide, busy, progress, documents, onCreate }: { guide?: StudyGuide; busy: boolean; progress?: { progress: number; message: string }; documents: Doc[]; onCreate: (doc: Doc) => void }) {
  const contentRef = useRef<HTMLDivElement>(null);
  function toggleAllSections() {
    const sections = [...(contentRef.current?.querySelectorAll<HTMLDetailsElement>("details") || [])];
    const shouldOpen = !sections.every((section) => section.open);
    sections.forEach((section) => { section.open = shouldOpen; });
  }
  function jumpToSection(index: number) {
    const container = contentRef.current;
    const target = container?.querySelector<HTMLElement>(`#study-section-${index}`);
    if (container && target) {
      const top = target.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
      container.scrollTo({ top, behavior: "smooth" });
    }
  }
  if (busy) return <div className="study-guide-page"><div className="study-guide-loading"><LoaderCircle className="spin" size={24} /><h2>{progress?.message || "Preparing study notes"}</h2><p>Reading the presentation in slide order. Larger decks can take a few minutes.</p><div className="progress-track"><span style={{ width: `${progress?.progress || 3}%` }} /></div><small>{progress?.progress || 3}%</small></div></div>;
  if (!guide) {
    const presentations = documents.filter((doc) => /\.pptx(?:\.pdf)?$/i.test(doc.filename));
    return <div className="study-guide-page"><div className="study-guide-empty"><BookOpen size={24} /><h2>Build notes from a presentation</h2><p>Studywise will read every slide in order and build a detailed, navigable guide.</p>{presentations.length ? <div className="study-guide-picker">{presentations.map((doc) => <button key={doc.id} onClick={() => onCreate(doc)}><FileText size={15} /><span>{doc.filename}</span><Sparkles size={15} /></button>)}</div> : <small>Upload a PowerPoint presentation to get started.</small>}</div></div>;
  }
  return <div className="study-guide-page">
    <div className="study-guide-heading"><div className="eyebrow">DETAILED STUDY NOTES</div><h1>{guide.filename}</h1><p>{guide.coverage_complete ? `Processed all ${guide.slides_processed} of ${guide.slide_count} slides.` : `Processed ${guide.slides_processed} of ${guide.slide_count} slides.`}</p>{guide.empty_slides.length > 0 && <small>No readable text or indexed visuals were found on {guide.empty_slides.length} slide{guide.empty_slides.length === 1 ? "" : "s"}.</small>}</div>
    <div className="study-guide-layout">
      <aside className="study-guide-outline"><div className="study-outline-title">IN THIS GUIDE</div><button className="study-toggle-all" onClick={toggleAllSections}>Expand / collapse all</button>{guide.sections.map((section, index) => <button key={index} onClick={() => jumpToSection(index)}>{String(index + 1).padStart(2, "0")} · {section.title}</button>)}</aside>
      <div className="study-guide-content" ref={contentRef}>{guide.sections.map((section, index) => <StudyNoteSection section={section} index={index} figures={guide.figures || []} key={index} />)}</div>
    </div>
  </div>;
}

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [activeTab, setActiveTab] = useState<"chat" | "quiz" | "study">("chat");
  const [quizTopic, setQuizTopic] = useState("");
  const [quiz, setQuiz] = useState<{ question: string; answer: string; evidence: string }[]>([]);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [showAnswers, setShowAnswers] = useState(false);
  const [studyGuide, setStudyGuide] = useState<StudyGuide>();
  const [taskProgress, setTaskProgress] = useState<{ progress: number; message: string }>();
  const fileInput = useRef<HTMLInputElement>(null);
  const conversationPane = useRef<HTMLDivElement>(null);

  async function refreshDocs() {
    try { const res = await fetch(`${api}/documents`); if (res.ok) setDocs(await apiJson<Doc[]>(res, "Loading documents")); }
    catch { setMessage("Can't reach the API. Start the Docker Compose stack and retry."); }
  }
  async function refreshChatSessions() {
    try { const res = await fetch(`${api}/sessions`); if (res.ok) setChatSessions(await apiJson<ChatSession[]>(res, "Loading chat history")); }
    catch { /* Keep the current chat usable when history cannot be loaded. */ }
  }
  async function openChat(session: ChatSession) {
    try {
      const res = await fetch(`${api}/sessions/${session.session_id}`);
      const saved = await apiJson<{ session_id: string; messages: { role: "user" | "assistant"; content: string; structured_answer?: StructuredAnswer | null }[] }>(res, "Loading conversation");
      setSessionId(saved.session_id);
      setTurns(saved.messages.map((item) => ({ role: item.role, text: item.content, structured: item.structured_answer || undefined })));
      setQuestion(""); setQuiz([]); setMessage(""); setActiveTab("chat");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not load this conversation"); }
  }
  useEffect(() => { void refreshDocs(); void refreshChatSessions(); }, []);
  useEffect(() => {
    const pane = conversationPane.current;
    if (pane) pane.scrollTop = pane.scrollHeight;
  }, [turns]);

  async function waitForTask(taskId: string) {
    const deadline = Date.now() + 30 * 60 * 1000;
    while (Date.now() < deadline) {
      const res = await fetch(`${api}/tasks/${taskId}`);
      const task = await apiJson<BackgroundTask>(res, "Checking progress");
      setTaskProgress({ progress: task.progress, message: task.message });
      if (task.status === "completed") return task;
      if (task.status === "failed") throw new Error(task.error || task.message || "The task failed");
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
    throw new Error("This task is taking longer than expected. Check the API logs and try again.");
  }

  async function upload(file?: File) {
    if (!file) return;
    setBusy(true); setMessage(""); setTaskProgress({ progress: 1, message: "Uploading file" });
    const data = new FormData(); data.append("file", file);
    try {
      const task = await uploadWithProgress(data, (progress) => setTaskProgress({ progress, message: "Uploading file" }));
      const finished = await waitForTask(task.task_id);
      const doc = finished.document;
      if (!doc) throw new Error("Upload finished without returning the document details");
      await refreshDocs(); setMessage(`Added ${doc.filename} · ${doc.chunk_count} text passages and ${doc.figure_count || 0} figures indexed`); setShowUpload(false);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Upload failed"); }
    finally { setBusy(false); setTaskProgress(undefined); if (fileInput.current) fileInput.current.value = ""; }
  }

  async function generateStudyGuide(doc: Doc) {
    if (busy) return;
    setBusy(true); setMessage(""); setStudyGuide(undefined); setTaskProgress({ progress: 1, message: "Preparing all slides" }); setActiveTab("study");
    try {
      const response = await fetch(`${api}/documents/${doc.id}/study-guide`, { method: "POST" });
      const task = await apiJson<BackgroundTask>(response, "Could not start study notes");
      const finished = await waitForTask(task.task_id);
      if (!finished.study_guide) throw new Error("Study-note task completed without a guide");
      setStudyGuide(finished.study_guide);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Study-note generation failed");
      setStudyGuide(undefined);
    } finally { setBusy(false); setTaskProgress(undefined); }
  }

  async function removeDoc(id: string) {
    try { const res = await fetch(`${api}/documents/${id}`, { method: "DELETE" }); if (!res.ok) throw new Error("Could not remove this PDF"); await refreshDocs(); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Delete failed"); }
  }

  async function ask(event: FormEvent) {
    event.preventDefault(); const value = question.trim(); if (!value || busy) return;
    setQuestion(""); setTurns((prev) => [...prev, { role: "user", text: value }]); setBusy(true); setMessage("");
    try {
      const res = await fetch(`${api}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: value, session_id: sessionId, top_k: 12, explanation_level: "detailed" }) });
      const visualRequested = /\b(figure|diagram|chart|graph|image|illustrat(?:e|ion)|visual)\b/i.test(value);
      const body = await apiJson<{ answer: string; session_id: string; citations: Citation[]; figures?: Figure[]; structured_answer: StructuredAnswer }>(res, "Question failed");
      setSessionId(body.session_id); setTurns((prev) => [...prev, { role: "assistant", text: body.answer, citations: body.citations, figures: body.figures || [], structured: body.structured_answer, visualRequested }]);
      void refreshChatSessions();
    } catch (error) { setTurns((prev) => [...prev, { role: "assistant", text: error instanceof Error ? error.message : "I couldn't get an answer." }]); }
    finally { setBusy(false); }
  }

  async function makeQuiz(event: FormEvent) {
    event.preventDefault(); if (!quizTopic.trim() || busy) return;
    setBusy(true); setMessage(""); setShowAnswers(false);
    try {
      const res = await fetch(`${api}/tutor/quiz`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic: quizTopic, count: 5 }) });
      const body = await apiJson<{ questions: { question: string; answer: string; evidence: string }[] }>(res, "Quiz generation failed"); setQuiz(body.questions);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Quiz generation failed"); }
    finally { setBusy(false); }
  }

  function newSession() { setTurns([]); setQuiz([]); setSessionId(undefined); setMessage(""); setActiveTab("chat"); }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><GraduationCap size={21} /></div><span>studywise</span><span className="brand-dot">.</span></div>
      <button className="new-chat" onClick={newSession}><Plus size={16} /> New study session</button>
      <div className="side-label">YOUR WORKSPACE</div>
      <button className={`nav-item ${activeTab === "chat" ? "selected" : ""}`} onClick={() => setActiveTab("chat")}><MessageCircle size={17} /> Ask your materials</button>
      <button className={`nav-item ${activeTab === "quiz" ? "selected" : ""}`} onClick={() => setActiveTab("quiz")}><Sparkles size={17} /> Practice quiz</button>
      <button className={`nav-item ${activeTab === "study" ? "selected" : ""}`} onClick={() => setActiveTab("study")}><BookOpen size={17} /> Study notes</button>
      <div className="chat-history"><div className="side-label">RECENT CHATS</div>{chatSessions.length === 0 ? <p className="empty-library">Your conversations will appear here.</p> : chatSessions.map((session) => <button className={`history-item ${session.session_id === sessionId ? "active" : ""}`} key={session.session_id} title={session.title} onClick={() => void openChat(session)}><MessageCircle size={14} /><span>{session.title}</span></button>)}</div>
      <div className="library-head"><span className="side-label">YOUR LIBRARY</span><button aria-label="Add PDF" onClick={() => setShowUpload(true)}><Plus size={16} /></button></div>
      <div className="doc-list">
        {docs.length === 0 ? <p className="empty-library">Your course materials will live here.</p> : docs.map((doc) => <div className="doc-row" key={doc.id} title={doc.filename}><FileText size={15} /><span>{doc.filename}</span>{doc.filename.toLowerCase().endsWith(".pptx") || doc.filename.toLowerCase().endsWith(".pptx.pdf") ? <button className="doc-study" aria-label={`Create detailed notes from ${doc.filename}`} title="Create detailed study notes from every slide" disabled={busy} onClick={() => void generateStudyGuide(doc)}><Sparkles size={13} /></button> : null}<button className="doc-remove" aria-label={`Remove ${doc.filename}`} title="Remove material" onClick={() => void removeDoc(doc.id)}><Trash2 size={14} /></button></div>)}
      </div>
      <button className="upload-side" onClick={() => setShowUpload(true)}><Upload size={15} /> Add course material</button>
      <div className="sidebar-foot"><div className="avatar">S</div><div><b>Study space</b><small>Private workspace</small></div><BookOpen className="foot-icon" size={17} /></div>
    </aside>

    <main className="main-panel">
      <header className="topbar"><div><span className="crumb">Learning space</span><span className="slash">/</span><b>{activeTab === "chat" ? "AI tutor" : activeTab === "quiz" ? "Practice quiz" : "Study notes"}</b></div><div className="top-right"><span className="status-dot" /> Your materials only <button className="top-upload" onClick={() => setShowUpload(true)}><Plus size={15} /> Add PDF</button></div></header>
      <section className="content">
        {activeTab !== "study" ? (activeTab === "chat" ? <>
          <div ref={conversationPane} className={`conversation ${turns.length ? "has-turns" : ""}`}>
            {turns.length === 0 ? <div className="welcome">
              <div className="welcome-icon"><Sparkles size={22} /></div>
              <div className="eyebrow">YOUR PERSONAL COURSE TUTOR</div>
              <h1>Understand your<br /><em>course, deeply.</em></h1>
              <p>Ask questions about your notes and readings. I’ll find the relevant passages and explain them clearly, with sources you can check.</p>
              <div className="suggestions"><button onClick={() => setQuestion("Summarize the key ideas in my materials")}>Give me a summary <span>↗</span></button><button onClick={() => setQuestion("Explain a difficult concept simply")}>Explain a concept simply <span>↗</span></button><button onClick={() => setQuestion("What should I focus on for my exam?")}>Help me study for an exam <span>↗</span></button></div>
              {!docs.length && <button className="first-upload" onClick={() => setShowUpload(true)}><Upload size={16} /> Start by adding a course PDF</button>}
            </div> : <div className="turn-list">{turns.map((turn, index) => <article className={`turn ${turn.role}`} key={index}><div className="turn-avatar">{turn.role === "user" ? "S" : <Sparkles size={16} />}</div><div className="turn-body"><div className="turn-name">{turn.role === "user" ? "You" : "Studywise"}</div>{turn.structured ? <div className={`structured-answer ${turn.structured.abstained ? "is-abstained" : ""}`}>
              {turn.structured.abstained && <div className="answer-status">Not established by the retrieved material</div>}
              <div className="answer-intro"><FormattedText text={turn.structured.direct_answer.text} /></div>
              {!!turn.structured.sections?.length ? <div className="answer-study-sections">{turn.structured.sections.map((section, sectionIndex) => <section className="answer-study-section" key={sectionIndex}><h3>{section.title}</h3>{section.content.map((claim, claimIndex) => <div className="answer-detail-paragraph" key={claimIndex}><FormattedText text={claim.text} /></div>)}</section>)}</div> : <>
                {turn.structured.formula && <div className="answer-formula"><FormattedText text={formulaMarkup(turn.structured.formula.text)} /></div>}
                {!!turn.structured.formula_terms?.length && <div className="answer-formula-terms"><div>where:</div><ul>{turn.structured.formula_terms.map((term, termIndex) => <li key={termIndex}><FormattedText text={term.text} /></li>)}</ul></div>}
                {!!turn.structured.explanation.length && <div className="answer-detail-flow">{turn.structured.explanation.map((claim, claimIndex) => <div className="answer-detail-paragraph" key={claimIndex}><FormattedText text={claim.text} /></div>)}</div>}
              </>}
              {!!turn.structured.additional_context?.length && <section className="answer-additional-context"><h4>Additional context</h4>{turn.structured.additional_context.map((item, contextIndex) => <div key={contextIndex}><FormattedText text={item} /></div>)}</section>}
              {!!turn.figures?.some((figure) => turn.visualRequested || isFigureCited(figure, turn.structured)) && <section className="answer-visuals"><h4>Figure from your material</h4><div className="answer-figures">{turn.figures.filter((figure) => turn.visualRequested || isFigureCited(figure, turn.structured)).slice(0, 2).map((figure) => <figure className="answer-figure" key={figure.id}><img src={figure.image_url} alt={`Retrieved figure from ${figure.filename}, page or slide ${figure.page_number}`} loading="lazy" /><figcaption><button onClick={() => { const source = turn.citations?.find((citation) => citation.number === figure.number); if (source) setOpenCitation(source); }}>[{figure.number}] {figure.filename} · {figure.filename.toLowerCase().endsWith(".pptx") ? "slide" : "page"} {figure.page_number}</button></figcaption></figure>)}</div></section>}
              {turn.visualRequested && !turn.figures?.length && <section className="answer-visual-missing"><h4>Visual</h4><p>No relevant figure was indexed in the uploaded material for this answer.</p></section>}
              {!turn.structured.sections?.length && !!turn.structured.key_points.length && <section className="answer-takeaways"><h4>Key takeaways</h4><ul>{turn.structured.key_points.map((claim, claimIndex) => <li key={claimIndex}><span><FormattedText text={claim.text} /></span></li>)}</ul></section>}
              {!!turn.structured.limitations.length && <section className="answer-limitations"><h4>What the material doesn’t cover</h4>{turn.structured.limitations.map((item, limitationIndex) => <div key={limitationIndex}><FormattedText text={item} /></div>)}</section>}
            </div> : <div className="turn-text"><FormattedText text={turn.text} /></div>}</div></article>)}</div>}
          </div>
          <div className="composer-wrap"><form className="composer" onSubmit={ask}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(event as unknown as FormEvent); } }} placeholder="Ask anything about your course materials…" rows={2} disabled={busy} /><div className="composer-foot"><div><span className="context-pill"><BookOpen size={13} /> {docs.length ? `${docs.length} source${docs.length > 1 ? "s" : ""} connected` : "No sources yet"}</span><span className="hint">Grounded in your course materials</span></div><button className="send-btn" type="submit" disabled={busy || !question.trim()}>{busy ? <LoaderCircle className="spin" size={17} /> : <Send size={16} />} <span>Ask tutor</span></button></div></form><div className="privacy-note">Studywise can make mistakes. Check important details against your source material.</div></div>
        </> : <div className="quiz-page"><div className="quiz-heading"><div className="welcome-icon"><Sparkles size={22} /></div><div className="eyebrow">ACTIVE RECALL</div><h1>Practice what<br /><em>you’ve learned.</em></h1><p>Generate a short quiz from your course material. Try answering before revealing the key.</p></div><form className="quiz-form" onSubmit={makeQuiz}><label htmlFor="topic">What topic are you studying?</label><div className="quiz-input"><input id="topic" value={quizTopic} onChange={(e) => setQuizTopic(e.target.value)} placeholder="e.g. cellular respiration" /><button disabled={busy || !quizTopic.trim()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Create quiz</button></div></form>{quiz.length > 0 && <div className="quiz-results"><div className="quiz-results-head"><h2>{quiz.length} questions</h2><button onClick={() => setShowAnswers(!showAnswers)}>{showAnswers ? "Hide answers" : "Reveal answers"}</button></div>{quiz.map((item, i) => <article className="quiz-card" key={i}><div className="quiz-num">{String(i + 1).padStart(2, "0")}</div><div><h3>{item.question}</h3>{showAnswers && <div className="answer-reveal"><b>Answer guide</b><p>{item.answer}</p><small>{item.evidence}</small></div>}</div></article>)}</div>}</div>) : <StudyGuidePage guide={studyGuide} busy={busy && activeTab === "study"} progress={taskProgress} documents={docs} onCreate={(doc) => void generateStudyGuide(doc)} />}        {message && <div className="toast" role="status"><span>{message}</span><button onClick={() => setMessage("")} aria-label="Dismiss"><X size={15} /></button></div>}
      </section>
    </main>

    {showUpload && <div className="modal-backdrop" onClick={() => { if (!busy) setShowUpload(false); }}><section className="upload-modal" onClick={(event) => event.stopPropagation()}><button className="modal-close" onClick={() => !busy && setShowUpload(false)} aria-label="Close" disabled={busy}><X size={18} /></button><div className="welcome-icon"><Upload size={21} /></div><h2>Add course material</h2><p>Upload a PDF or PowerPoint. We’ll show progress while its text and figures are indexed.</p><input ref={fileInput} type="file" accept="application/pdf,.pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation,.pptx" onChange={(event) => void upload(event.target.files?.[0])} hidden /><button className="drop-zone" onClick={() => fileInput.current?.click()} disabled={busy}>{busy ? <><LoaderCircle className="spin" size={22} /><b>{taskProgress?.message || "Processing your file"}</b><div className="progress-track"><span style={{ width: `${taskProgress?.progress || 3}%` }} /></div><span>{taskProgress?.progress || 3}% complete</span></> : <><Upload size={22} /><b>Choose a PDF or PPTX file</b><span>PDF or PPTX · up to 25 MB</span></>}</button><div className="modal-safe"><Check size={14} /> Your files stay in this workspace</div></section></div>}

    {openCitation && <div className="modal-backdrop" onClick={() => setOpenCitation(null)}><section className="source-modal" onClick={(event) => event.stopPropagation()}><button className="modal-close" onClick={() => setOpenCitation(null)} aria-label="Close"><X size={18} /></button><div className="eyebrow">{openCitation.kind === "figure" ? "FIGURE" : "SOURCE"} {openCitation.number}</div><h2>{openCitation.filename}</h2><div className="source-page">Page or slide {openCitation.page}</div>{openCitation.image_url && <img className="source-figure-preview" src={openCitation.image_url} alt={`Figure from ${openCitation.filename}, page ${openCitation.page}`} />}<blockquote>{openCitation.excerpt}</blockquote><div className="source-score">Retrieved evidence · relevance {openCitation.score.toFixed(3)}</div></section></div>}
  </div>;
}
