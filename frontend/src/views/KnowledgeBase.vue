<template>
  <div class="kb-page">
    <!-- 左侧边栏 -->
    <aside class="kb-sidebar">
      <!-- 会话列表 -->
      <div class="sidebar-section">
        <div class="section-header">
          <span class="sidebar-title">对话</span>
          <button class="new-session-btn" @click="newSession">+ 新建</button>
        </div>
        <div class="session-list">
          <div
            v-for="sess in sessions"
            :key="sess.id"
            class="session-item"
            :class="{ active: sess.id === sessionId }"
            @click="switchSession(sess.id)"
          >
            <span class="session-title" :title="sess.title">{{ sess.title }}</span>
            <button
              class="session-delete-btn"
              @click.stop="handleDeleteSession(sess.id)"
              :disabled="sessions.length <= 1"
            >&#10005;</button>
          </div>
        </div>
      </div>

      <div class="sidebar-divider"></div>

      <!-- 知识库 -->
      <div class="sidebar-section">
        <span class="sidebar-title">知识库</span>

        <div class="upload-area">
          <input
            ref="fileInput"
            type="file"
            accept=".txt,.pdf"
            style="display:none"
            @change="handleUpload"
          />
          <button class="upload-btn" @click="$refs.fileInput.click()" :disabled="uploading">
            {{ uploading ? '上传中...' : '+ 上传文档' }}
          </button>
          <p class="upload-hint">支持 .txt / .pdf</p>
        </div>

        <div class="doc-list">
          <div class="doc-list-header">已上传 ({{ docs.length }})</div>
          <div v-if="docs.length === 0" class="doc-empty">暂无文档</div>
          <div v-for="doc in docs" :key="doc" class="doc-item">
            <span class="doc-icon">{{ doc.endsWith('.pdf') ? '📄' : '📝' }}</span>
            <span class="doc-name">{{ doc }}</span>
            <button class="doc-delete-btn" @click="deleteDoc(doc)" title="删除文档">&#10005;</button>
          </div>
        </div>

        <button class="refresh-btn" @click="loadDocs">刷新列表</button>
      </div>
    </aside>

    <!-- 右侧对话区 -->
    <main class="kb-chat">
      <div class="chat-header">
        <select v-model="mode" class="mode-select">
          <option value="rag">RAG 模式</option>
          <option value="agent">Agent 模式</option>
          <option value="reasoning">Agent 推理</option>
          <option value="langgraph">LangGraph Agent</option>
        </select>
        <span class="mode-hint">{{ modeLabelText() }}</span>
        <button class="clear-btn" @click="newSession" v-if="messages.length > 0">新建对话</button>
      </div>

      <div class="messages" ref="msgContainer">
        <div v-if="messages.length === 0" class="welcome">
          <div class="welcome-icon">📚</div>
          <div class="welcome-text">上传文档后开始提问</div>
        </div>

        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          class="message"
          :class="msg.role"
        >
          <div class="msg-label">{{ msg.label }}</div>
          <div class="msg-content">{{ msg.content }}</div>
          <div v-if="msg.streaming" class="cursor-blink">|</div>
        </div>

        <div v-if="errorMsg" class="chat-error">{{ errorMsg }}</div>
      </div>

      <div class="input-area">
        <div class="input-wrapper">
          <div class="input-row">
            <input
              v-model="question"
              class="chat-input"
              placeholder="输入问题，回车发送..."
              :disabled="loading"
              @keyup.enter="send"
            />
            <button class="send-btn" :disabled="loading || !question.trim()" @click="send">
              {{ loading ? '...' : '发送' }}
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, watch } from 'vue'
import axios from 'axios'

const API_BASE = 'http://127.0.0.1:8010'

const ENDPOINTS = {
  rag: '/api/kb/chat',
  agent: '/api/kb/chat/agent',
  reasoning: '/api/kb/chat/agent/reasoning',
  langgraph: '/api/kb/chat/agent/langgraph',
}

const MODE_LABELS = {
  rag: '固定检索 → 回答',
  agent: '模型自主决定是否检索（单轮）',
  reasoning: '多轮推理，可连续检索多次',
  langgraph: 'LangGraph 图结构 Agent',
}

// ============================================================
// 会话管理（localStorage）
// ============================================================
const SESSIONS_LIST_KEY = 'kb_sessions_list'

function getSessionsList() {
  try {
    return JSON.parse(localStorage.getItem(SESSIONS_LIST_KEY) || '[]')
  } catch { return [] }
}

function saveSessionsList(list) {
  localStorage.setItem(SESSIONS_LIST_KEY, JSON.stringify(list))
}

function loadMessages(sid) {
  try {
    return JSON.parse(localStorage.getItem('kb_msgs_' + sid) || '[]')
  } catch { return [] }
}

function saveMessages(sid, msgs) {
  const clean = msgs.map(({ role, label, content }) => ({ role, label, content }))
  localStorage.setItem('kb_msgs_' + sid, JSON.stringify(clean))
}

function autoTitle(messages) {
  const firstUser = messages.find(m => m.role === 'user')
  return firstUser ? firstUser.content.slice(0, 20) : '新对话'
}

// ============================================================
// 上传相关
// ============================================================
const fileInput = ref(null)
const uploading = ref(false)
const docs = ref([])

const loadDocs = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/kb/documents`)
    if (res.data.code === 200) {
      docs.value = res.data.data.documents
    }
  } catch (e) {
    console.error('加载文档列表失败:', e)
    errorMsg.value = '加载文档列表失败，请检查后端服务'
  }
}

const deleteDoc = async (filename) => {
  try {
    const res = await axios.delete(`${API_BASE}/api/kb/documents/${encodeURIComponent(filename)}`)
    if (res.data.code === 200) {
      await loadDocs()
    } else {
      errorMsg.value = res.data.message || '删除失败'
    }
  } catch (err) {
    errorMsg.value = '删除失败，请检查后端服务'
  }
}

const handleUpload = async (e) => {
  const file = e.target.files[0]
  if (!file) return

  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('file', file)
    const res = await axios.post(`${API_BASE}/api/kb/upload`, formData)
    if (res.data.code === 200) {
      await loadDocs()
    } else {
      errorMsg.value = res.data.message || '上传失败'
    }
  } catch (err) {
    errorMsg.value = err.response?.data?.detail || '上传失败，请检查后端服务'
  } finally {
    uploading.value = false
    e.target.value = ''
  }
}

// ============================================================
// 对话
// ============================================================
const mode = ref('rag')
const question = ref('')
const loading = ref(false)
const errorMsg = ref('')
const messages = ref([])
const msgContainer = ref(null)
const sessionId = ref('')
const sessions = ref([])

// 初始化会话列表
function initSessions() {
  sessions.value = getSessionsList()
  if (sessions.value.length === 0) {
    // 首次使用，创建一个默认会话
    const sid = 'session_' + Date.now()
    sessions.value = [{ id: sid, title: '新对话' }]
    saveSessionsList(sessions.value)
  }
  sessionId.value = sessions.value[0].id
  messages.value = loadMessages(sessionId.value)
}

function newSession() {
  const sid = 'session_' + Date.now()
  sessions.value.unshift({ id: sid, title: '新对话' })
  saveSessionsList(sessions.value)
  sessionId.value = sid
  messages.value = []
}

function switchSession(sid) {
  if (sid === sessionId.value) return
  sessionId.value = sid
  messages.value = loadMessages(sid)
  errorMsg.value = ''
}

function handleDeleteSession(sid) {
  if (sessions.value.length <= 1) return
  const idx = sessions.value.findIndex(s => s.id === sid)
  sessions.value.splice(idx, 1)
  saveSessionsList(sessions.value)
  localStorage.removeItem('kb_msgs_' + sid)

  if (sid === sessionId.value) {
    sessionId.value = sessions.value[0].id
    messages.value = loadMessages(sessionId.value)
  }
}

// 消息变化时自动存
watch(messages, (val) => {
  const sid = sessionId.value
  saveMessages(sid, val)
  // 更新会话标题
  const list = getSessionsList()
  const entry = list.find(s => s.id === sid)
  if (entry) {
    entry.title = autoTitle(val)
    saveSessionsList(list)
    sessions.value = list
  }
}, { deep: true })

const modeLabelText = () => MODE_LABELS[mode.value]

const scrollToBottom = async () => {
  await nextTick()
  const el = msgContainer.value
  if (el) el.scrollTop = el.scrollHeight
}

async function readStream(response, aiMsg) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    aiMsg.content += decoder.decode(value, { stream: true })
    await scrollToBottom()
  }
}

const send = async () => {
  const q = question.value.trim()
  if (!q || loading.value) return

  errorMsg.value = ''
  question.value = ''

  messages.value.push({ role: 'user', label: '用户', content: q })
  await scrollToBottom()

  loading.value = true

  messages.value.push({ role: 'ai', label: modeLabelText(), content: '', streaming: true })
  const aiMsg = messages.value[messages.value.length - 1]
  await scrollToBottom()

  try {
    const response = await fetch(`${API_BASE}${ENDPOINTS[mode.value]}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: sessionId.value }),
    })
    await readStream(response, aiMsg)
  } catch (err) {
    errorMsg.value = err.message || '请求失败，请检查后端服务是否正常运行'
    messages.value.pop()
  } finally {
    aiMsg.streaming = false
    loading.value = false
  }
}

onMounted(() => {
  initSessions()
  loadDocs()
})
</script>

<style scoped>
.kb-page {
  display: flex;
  height: calc(100vh - 60px);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* ===== 左侧边栏 ===== */
.kb-sidebar {
  width: 260px;
  background: #f8f9fa;
  border-right: 1px solid #e8e8e8;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 0;
  overflow-y: auto;
}

.sidebar-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.sidebar-title {
  font-size: 16px;
  font-weight: 700;
  color: #1a1a1a;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.new-session-btn {
  background: transparent;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  color: #ff3b3b;
  cursor: pointer;
  transition: all 0.2s;
}

.new-session-btn:hover {
  background: #ff3b3b;
  color: white;
  border-color: #ff3b3b;
}

/* 会话列表 */
.session-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 240px;
  overflow-y: auto;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}

.session-item:hover {
  background: #e8e8e8;
}

.session-item.active {
  background: #e0e0ff;
}

.session-title {
  flex: 1;
  font-size: 13px;
  color: #333;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-delete-btn {
  background: none;
  border: none;
  font-size: 12px;
  color: #ccc;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 4px;
  flex-shrink: 0;
  line-height: 1;
}

.session-delete-btn:hover:not(:disabled) {
  color: #ff3b3b;
  background: #fff0f0;
}

.session-delete-btn:disabled {
  opacity: 0;
  cursor: default;
}

.sidebar-divider {
  height: 1px;
  background: #e0e0e0;
  margin: 16px 0;
}

/* 上传 */
.upload-area {
  text-align: center;
}

.upload-btn {
  width: 100%;
  padding: 10px 0;
  background: #ff3b3b;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s;
}

.upload-btn:hover:not(:disabled) {
  background: #e53030;
}

.upload-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.upload-hint {
  font-size: 12px;
  color: #999;
  margin: 6px 0 0;
}

/* 文档列表 */
.doc-list {
  overflow-y: auto;
  max-height: 160px;
}

.doc-list-header {
  font-size: 13px;
  font-weight: 600;
  color: #555;
  margin-bottom: 8px;
}

.doc-empty {
  font-size: 13px;
  color: #bbb;
  text-align: center;
  padding: 10px 0;
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 13px;
  color: #333;
  word-break: break-all;
}

.doc-item:hover {
  background: #eee;
}

.doc-item:hover .doc-delete-btn {
  opacity: 1;
}

.doc-delete-btn {
  background: none;
  border: none;
  font-size: 12px;
  color: #ccc;
  cursor: pointer;
  padding: 2px 4px;
  opacity: 0;
  transition: opacity 0.15s;
  flex-shrink: 0;
  line-height: 1;
  border-radius: 4px;
}

.doc-delete-btn:hover {
  color: #ff3b3b;
  background: #fff0f0;
}

.doc-icon {
  font-size: 14px;
  flex-shrink: 0;
}

.refresh-btn {
  width: 100%;
  padding: 8px 0;
  background: transparent;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 13px;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.refresh-btn:hover {
  background: #eee;
  color: #333;
}

/* ===== 右侧对话区 ===== */
.kb-chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
}

.chat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 20px;
  border-bottom: 1px solid #f0f0f0;
}

.mode-select {
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  background: white;
  outline: none;
  cursor: pointer;
}

.mode-select:focus {
  border-color: #ff3b3b;
}

.mode-hint {
  font-size: 12px;
  color: #999;
  flex: 1;
}

.clear-btn {
  padding: 4px 12px;
  background: transparent;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 12px;
  color: #999;
  cursor: pointer;
}

.clear-btn:hover {
  color: #ff3b3b;
  border-color: #ff3b3b;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.welcome {
  text-align: center;
  margin-top: 120px;
}

.welcome-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.welcome-text {
  font-size: 15px;
  color: #999;
}

.message {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
}

.message.user {
  align-self: flex-end;
  background: #ff3b3b;
  color: white;
}

.message.ai {
  align-self: flex-start;
  background: #f0f0f0;
  color: #333;
}

.msg-label {
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
  opacity: 0.7;
}

.msg-content {
  white-space: pre-wrap;
  word-break: break-word;
}

.cursor-blink {
  display: inline;
  animation: blink 0.6s infinite;
  color: #ff3b3b;
  font-weight: bold;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.chat-error {
  align-self: center;
  background: #fff0f0;
  color: #ff3b3b;
  font-size: 13px;
  padding: 10px 16px;
  border-radius: 8px;
}

/* ===== 底部输入 ===== */
.input-area {
  display: flex;
  padding: 14px 20px;
  border-top: 1px solid #f0f0f0;
}

.input-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.input-row {
  display: flex;
  gap: 10px;
}

.chat-input {
  flex: 1;
  height: 44px;
  border: 1px solid #e0e0e0;
  border-radius: 22px;
  padding: 0 20px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: #ff3b3b;
}

.send-btn {
  height: 44px;
  width: 80px;
  background: #ff3b3b;
  color: white;
  border: none;
  border-radius: 22px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s;
}

.send-btn:hover:not(:disabled) {
  background: #e53030;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
