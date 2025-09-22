const menuBtn = document.getElementById("menuBtn");
const sidebar = document.getElementById("sidebar");
const closeBtn = document.getElementById("closeBtn");
const chatContainer = document.getElementById("chatContainer");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChat");
const historyContainer = document.getElementById("history");

let chatHistory = [];
let currentChat = [];

// Sidebar toggle
menuBtn.addEventListener("click", () => {
  sidebar.classList.add("open");
  menuBtn.style.display = "none";
});
closeBtn.addEventListener("click", () => {
  sidebar.classList.remove("open");
  menuBtn.style.display = "block";
});

// Add message
function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.classList.add("message", type);
  msg.innerText = text;
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  currentChat.push({ text, type });
}

// Send message
sendBtn.addEventListener("click", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  const placeholder = document.querySelector(".placeholder");
  if (placeholder) placeholder.remove();

  addMessage(text, "user");
  chatInput.value = "";

  //Send to backend
  fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: text})
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      addMessage("Error: " + data.error, "bot");
    } else {
      addMessage(data.answer, "bot");
      // addMessage(`Retrieved ${data.retrieval_count} passages.`, "bot");
    }
  })
  .catch(error => addMessage("Error: Failed to get response.", "bot"));
});

// New Chat
newChatBtn.addEventListener("click", () => {
  if (currentChat.length > 0) {
    const chatId = "Chat " + (chatHistory.length + 1);
    chatHistory.push({ id: chatId, messages: [...currentChat] });

    const item = document.createElement("div");
    item.classList.add("history-item");
    item.innerText = chatId;
    item.addEventListener("click", () => loadChat(chatId));

    if (historyContainer.innerText === "No chats yet") historyContainer.innerHTML = "";
    historyContainer.appendChild(item);
  }

  currentChat = [];
  chatContainer.innerHTML = '<p class="placeholder">Ask a question about lecture transcripts...</p>';
});

// Load chat
function loadChat(chatId) {
  const chat = chatHistory.find(c => c.id === chatId);
  if (!chat) return;
  chatContainer.innerHTML = "";
  chat.messages.forEach(msg => addMessage(msg.text, msg.type));
  currentChat = [...chat.messages];
}

//Add event listener for "Enter" key on the input field
chatInput.addEventListener("keydown", (e) => {
    if (e.key == "Enter"){
        e.preventDefault();
        sendBtn.click();
    }
});