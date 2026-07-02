const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const scroller = document.getElementById("messages");
const column = document.getElementById("column");
const emptyState = document.getElementById("empty-state");

const SPARK_ICON =
  '<svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">' +
  '<path d="M8 0c.5 4.2 3.3 7 7.5 7.5v1C11.3 9 8.5 11.8 8 16c-.5-4.2-3.3-7-7.5-7.5v-1C4.7 7 7.5 4.2 8 0z"/></svg>';

let conversationId = null;
let streaming = false;
let controller = null;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (streaming) {
    controller?.abort(); // the send button doubles as Stop while streaming
    return;
  }
  send();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});

input.addEventListener("input", autosize);

document.querySelectorAll(".suggestions .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.textContent;
    send();
  });
});

async function send() {
  const text = input.value.trim();
  if (!text || streaming) return;

  controller = new AbortController();
  setStreaming(true);
  emptyState.hidden = true;
  input.value = "";
  autosize();
  addUserMessage(text);
  const assistant = addAssistantMessage();

  try {
    await streamReply(text, assistant);
  } catch (error) {
    if (error.name !== "AbortError") {
      assistant.text.textContent = "Something went wrong. Check the server and try again.";
      assistant.text.classList.add("reply-error");
    }
    // Stopped on purpose: keep the partial text; the server drops it from history.
  } finally {
    controller = null;
    assistant.text.classList.remove("streaming");
    setStreaming(false);
    input.focus();
  }
}

async function streamReply(text, assistant) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, conversation_id: conversationId }),
    signal: controller.signal,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // partial NDJSON line waits for the next chunk
    for (const line of lines) {
      if (line.trim()) handleEvent(JSON.parse(line), assistant);
    }
  }
  if (buffer.trim()) handleEvent(JSON.parse(buffer), assistant);
}

function handleEvent(event, assistant) {
  switch (event.type) {
    case "message_start":
      if (event.conversation_id) conversationId = event.conversation_id;
      break;
    case "text_delta":
      assistant.text.textContent += event.text;
      stickToBottom();
      break;
    case "tool_start":
      setToolChip(assistant, "running", `Running ${event.tool_name}`, null);
      break;
    case "tool_result":
      setToolChip(assistant, "ok", event.tool_name, event.result);
      break;
    case "tool_error":
      setToolChip(assistant, "failed", event.tool_name, event.error);
      break;
    case "error":
      assistant.text.textContent = "The assistant hit an internal error.";
      assistant.text.classList.add("reply-error");
      break;
    case "message_done":
      break;
  }
}

function addUserMessage(text) {
  const row = document.createElement("article");
  row.className = "msg user";
  const body = document.createElement("div");
  body.className = "msg-text";
  body.dir = "auto"; // pick LTR/RTL from content so Arabic, Hebrew, etc. read correctly
  body.textContent = text; // textContent everywhere: user/model text is never HTML
  row.appendChild(body);
  column.appendChild(row);
  stickToBottom(true);
}

function addAssistantMessage() {
  const row = document.createElement("article");
  row.className = "msg assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = SPARK_ICON; // static markup constant, never user data

  const body = document.createElement("div");
  body.className = "msg-body";
  const chip = document.createElement("div");
  chip.className = "tool-chip";
  chip.hidden = true;
  const dot = document.createElement("span");
  dot.className = "dot";
  const label = document.createElement("span");
  const detail = document.createElement("code");
  detail.dir = "auto"; // tool output may echo a place name in any script
  chip.append(dot, label, detail);

  const text = document.createElement("div");
  text.className = "msg-text streaming";
  text.dir = "auto"; // reply direction follows the language the model streams

  body.append(chip, text);
  row.append(avatar, body);
  column.appendChild(row);
  stickToBottom(true);
  return { chip, label, detail, text };
}

function setToolChip(assistant, state, labelText, detailText) {
  assistant.chip.hidden = false;
  assistant.chip.className = `tool-chip ${state}`;
  assistant.label.textContent = labelText;
  assistant.detail.textContent = detailText ?? "";
  stickToBottom();
}

function setStreaming(value) {
  streaming = value;
  input.disabled = value;
  sendButton.classList.toggle("stop", value);
  sendButton.setAttribute("aria-label", value ? "Stop the reply" : "Send message");
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 168)}px`;
}

function stickToBottom(force = false) {
  const nearBottom =
    scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
  if (force || nearBottom) scroller.scrollTop = scroller.scrollHeight;
}
