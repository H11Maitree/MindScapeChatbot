const chatContainer = document.getElementById("messages");
const input = document.getElementById("userInput");
const sendButton = document.querySelector("button");

// ฟังก์ชันช่วยแสดงข้อความ
function appendMessage(sender, message) {
  const msg = document.createElement("p");
  msg.innerHTML = `<strong>${sender}:</strong> ${message}`;
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// เมื่อผู้ใช้กดส่ง
async function sendMessage() {
  const userMessage = input.value.trim();
  if (!userMessage) return;

  appendMessage("คุณ", userMessage);
  input.value = "";

  // ตรวจคำสำคัญทางธรรมะ
  const dhammaKeywords = ["อริยสัจ", "หลักธรรม", "นิพพาน"];
  const isDhamma = dhammaKeywords.some(w => userMessage.includes(w));

  // เลือก endpoint ตามประเภทข้อความ
  const endpoint = isDhamma ? "/ask" : "/chat";
  try {
    const response = await fetch(`${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage })
    });
    const data = await response.json();
    appendMessage("Chatbot", data.reply);
  } catch (error) {
    console.error("❌ Error:", error);
    appendMessage("Chatbot", "เกิดข้อผิดพลาดในการเชื่อมต่อ");
  }
}

// ผูกอีเวนต์
sendButton.addEventListener("click", sendMessage);
input.addEventListener("keypress", e => {
  if (e.key === "Enter") sendMessage();
});
