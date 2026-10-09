const messagesEl = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const companySelect = document.getElementById('companySelect');
const bboxHighlight = document.getElementById('bboxHighlight');
const metaBox = document.getElementById('metaBox');
const pdfFrame = document.getElementById('pdfFrame');
const fakeDocument = document.getElementById('fakeDocument');

const DEMO_BBOX = [0.142, 0.098, 0.915, 0.133];

function addMessage(text, role = 'assistant') {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  msg.textContent = text;
  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setHighlight(bbox) {
  if (!bbox || bbox.length !== 4) {
    bboxHighlight.style.display = 'none';
    return;
  }

  const [ymin, xmin, ymax, xmax] = bbox;
  const left = xmin * 100;
  const top = ymin * 100;
  const width = Math.max((xmax - xmin) * 100, 2);
  const height = Math.max((ymax - ymin) * 100, 2);

  bboxHighlight.style.display = 'block';
  bboxHighlight.style.left = `${left}%`;
  bboxHighlight.style.top = `${top}%`;
  bboxHighlight.style.width = `${width}%`;
  bboxHighlight.style.height = `${height}%`;
}

async function loadPdfViewer(company = 'VNM', year = 2025) {
  const pdfUrl = `/api/v1/pdf/${company}/${year}`;
  try {
    const response = await fetch(pdfUrl, { method: 'GET' });
    const contentType = response.headers.get('content-type') || '';
    if (response.ok && contentType.includes('application/pdf')) {
      pdfFrame.src = pdfUrl;
      pdfFrame.style.display = 'block';
      fakeDocument.style.display = 'none';
      return;
    }
  } catch (error) {
    console.warn('PDF not available, falling back to demo document', error);
  }

  pdfFrame.src = '';
  pdfFrame.style.display = 'none';
  fakeDocument.style.display = 'block';
}

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  const company = companySelect.value;
  addMessage(query, 'user');
  addMessage('Đang xử lý...', 'assistant');

  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        company,
        year: 2025,
        thread_id: 'ui-demo-thread'
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assistantText = '';
    let metadata = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop() || '';

      for (const chunk of chunks) {
        const eventMatch = chunk.match(/event:\s*(.+)/);
        const dataMatch = chunk.match(/data:\s*(.*)/s);
        const event = eventMatch ? eventMatch[1].trim() : 'message';
        const data = dataMatch ? dataMatch[1].trim() : '';

        if (event === 'token' && data) {
          assistantText += data;
          const last = messagesEl.lastElementChild;
          if (last && last.classList.contains('assistant')) {
            last.textContent = assistantText;
          }
        }

        if (event === 'metadata' && data) {
          try {
            metadata = JSON.parse(data);
          } catch (error) {
            console.error('Failed to parse metadata', error);
          }
        }
      }
    }

    if (metadata && metadata.citations && metadata.citations.length > 0) {
      const first = metadata.citations[0];
      const bbox = first.bbox || first.bounding_box || null;
      setHighlight(bbox);
      metaBox.textContent = `Trích dẫn: page ${first.page ?? 1} · bbox ${bbox ? JSON.stringify(bbox) : 'n/a'} · source: ${first.source_type || 'unknown'}`;
    } else {
      const fallbackBbox = DEMO_BBOX;
      setHighlight(fallbackBbox);
      metaBox.textContent = `Trích dẫn mẫu: page 12 · bbox ${JSON.stringify(fallbackBbox)}`;
    }
  } catch (error) {
    const last = messagesEl.lastElementChild;
    if (last && last.classList.contains('assistant')) {
      last.textContent = 'Không thể kết nối API. Hãy kiểm tra backend đang chạy trên http://127.0.0.1:8000.';
    }
    console.error(error);
  }
}

sendBtn.addEventListener('click', sendQuery);
queryInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    sendQuery();
  }
});

loadPdfViewer('VNM', 2025);
setHighlight(DEMO_BBOX);
metaBox.textContent = `Trích dẫn mẫu: page 12 · bbox ${JSON.stringify(DEMO_BBOX)}`;
