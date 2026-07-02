const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");

let conversationId = null;
let streaming = false;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || streaming) return;

  setStreaming(true);
  input.value = "";
  addUserBubble(text);
  const assistant = addAssistantBubble();

  try {
    await streamReply(text, assistant);
  } catch {
    assistant.text.textContent = "Something went wrong. Please try again.";
    assistant.bubble.classList.add("error-bubble");
  } finally {
    setStreaming(false);
    input.focus();
  }
});

async function streamReply(text, assistant) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, conversation_id: conversationId }),
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
      scrollToBottom();
      break;
    case "tool_start":
      setToolStatus(assistant, "running", `${event.tool_name} running…`);
      break;
    case "tool_result":
      setToolStatus(assistant, "ok", `${event.tool_name}: ${event.result}`);
      break;
    case "tool_error":
      setToolStatus(assistant, "failed", `${event.tool_name}: ${event.error}`);
      break;
    case "error":
      assistant.text.textContent = "The assistant hit an internal error.";
      assistant.bubble.classList.add("error-bubble");
      break;
    case "message_done":
      break;
  }
}

function addUserBubble(text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = text; // textContent everywhere: user/model text is never HTML
  messages.appendChild(bubble);
  scrollToBottom();
}

function addAssistantBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  const status = document.createElement("span");
  status.className = "tool-status";
  const text = document.createElement("div");
  bubble.append(status, text);
  messages.appendChild(bubble);
  scrollToBottom();
  return { bubble, status, text };
}

function setToolStatus(assistant, state, label) {
  assistant.status.className = `tool-status ${state}`;
  assistant.status.textContent = label;
}

function setStreaming(value) {
  streaming = value;
  input.disabled = value;
  sendButton.disabled = value;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}
