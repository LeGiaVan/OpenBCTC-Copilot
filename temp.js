
        // --- State ---
        let currentCitations = [];
        let pdfDoc = null;
        let pageNum = 1;
        let pageRendering = false;
        let pageNumPending = null;
        const scale = 1.5;
        const canvas = document.getElementById('pdf-canvas');
        const ctx = canvas.getContext('2d');
        // Hỗ trợ chạy tách riêng Frontend (Live Server, file://)
        const API_BASE_URL = (window.location.origin.includes('8000') || window.location.protocol === 'file:') 
            ? 'http://127.0.0.1:8000' 
            : window.location.origin.includes('127.0.0.1') || window.location.origin.includes('localhost') 
                ? 'http://127.0.0.1:8000' 
                : '';

        // Global State
        let CURRENT_COMPANY = 'VNM';
        let CURRENT_YEAR = 2025;

        // Fetch danh sách Documents
        async function loadDocuments() {
            try {
                const res = await fetch(API_BASE_URL + '/api/v1/documents');
                const data = await res.json();
                const listEl = document.getElementById('doc-list');
                listEl.innerHTML = '';
                
                if (data.documents.length === 0) {
                    listEl.innerHTML = '<div class="text-center text-sm text-gray-500">Chưa có dữ liệu nào trong thư mục data/</div>';
                    return;
                }

                data.documents.forEach(doc => {
                    const btn = document.createElement('button');
                    btn.className = "w-full text-left px-4 py-3 rounded-lg border border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-all flex justify-between items-center group";
                    btn.innerHTML = `
                        <div>
                            <div class="font-semibold text-gray-800 group-hover:text-blue-700">${doc.name}</div>
                            <div class="text-xs text-gray-400 mt-1">ID: ${doc.id}</div>
                        </div>
                        <div class="text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity">➔</div>
                    `;
                    btn.onclick = () => selectDocument(doc);
                    listEl.appendChild(btn);
                });
            } catch (error) {
                console.error("Error loading documents:", error);
                document.getElementById('doc-list').innerHTML = '<div class="text-center text-sm text-red-500">Lỗi kết nối đến Backend</div>';
            }
        }

        // Chọn Document và vào App
        function selectDocument(doc) {
            CURRENT_COMPANY = doc.company;
            CURRENT_YEAR = doc.year;
            
            // Cập nhật UI (Fade out selector, fade in panes)
            document.getElementById('doc-selector').classList.add('opacity-0', 'pointer-events-none');
            setTimeout(() => {
                document.getElementById('doc-selector').classList.add('hidden');
                
                const chatPane = document.getElementById('main-chat-pane');
                const pdfPane = document.getElementById('main-pdf-pane');
                
                chatPane.classList.remove('opacity-0', 'pointer-events-none');
                pdfPane.classList.remove('opacity-0', 'pointer-events-none');
            }, 300);

            // Cập nhật text trên Header
            const headerBadge = document.querySelector('.bg-blue-900');
            if(headerBadge) headerBadge.textContent = `${doc.company} - ${doc.year}`;
            
            const pdfLabel = document.querySelector('.text-red-500').nextSibling;
            if(pdfLabel) pdfLabel.textContent = ` ${doc.id}.pdf`;
            
            // Tải file PDF
            loadPDF();
        }

        // --- Init PDF.js ---
        function loadPDF() {
            const url = API_BASE_URL + `/api/v1/pdf/${CURRENT_COMPANY.toLowerCase()}/${CURRENT_YEAR}?t=` + new Date().getTime();
            
            pdfjsLib.getDocument({
                url: url,
            cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
            cMapPacked: true,
            standardFontDataUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/standard_fonts/'
        }).promise.then(function(pdfDoc_) {
            pdfDoc = pdfDoc_;
            document.getElementById('page-count').textContent = pdfDoc.numPages;
            renderPage(pageNum);
        }).catch(err => {
            console.error("Lỗi tải PDF:", err);
            ctx.fillStyle = "#f3f4f6";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#ef4444";
            ctx.font = "16px sans-serif";
            ctx.fillText("Không tìm thấy file PDF trên server", 50, 50);
        });

        function renderPage(num) {
            pageRendering = true;
            pdfDoc.getPage(num).then(function(page) {
                // Tự động scale PDF cho vừa khít chiều rộng của khung bên phải
                const container = document.getElementById('pdf-container');
                const unscaledViewport = page.getViewport({scale: 1.0});
                const dynamicScale = (container.clientWidth - 40) / unscaledViewport.width; // 40px padding
                
                const viewport = page.getViewport({scale: dynamicScale});
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                bboxLayer.style.width = viewport.width + 'px';
                bboxLayer.style.height = viewport.height + 'px';
                bboxLayer.innerHTML = ''; // Clear old bboxes

                const renderContext = {
                    canvasContext: ctx,
                    viewport: viewport
                };
                const renderTask = page.render(renderContext);

                renderTask.promise.then(function() {
                    pageRendering = false;
                    if (pageNumPending !== null) {
                        renderPage(pageNumPending);
                        pageNumPending = null;
                    }
                });
            });
            document.getElementById('page-num').textContent = num;
        }

        function queueRenderPage(num) {
            if (pageRendering) {
                pageNumPending = num;
            } else {
                renderPage(num);
            }
        }

        document.getElementById('prev-page').addEventListener('click', () => {
            if (pageNum <= 1) return;
            pageNum--;
            queueRenderPage(pageNum);
        });

        document.getElementById('next-page').addEventListener('click', () => {
            if (pageNum >= pdfDoc.numPages) return;
            pageNum++;
            queueRenderPage(pageNum);
        });

        // Tự động resize lại PDF khi thay đổi kích thước cửa sổ
        window.addEventListener('resize', () => {
            if (pdfDoc) queueRenderPage(pageNum);
        });

        // --- Chat Logic ---
        const chatForm = document.getElementById('chat-form');
        const queryInput = document.getElementById('query-input');
        const chatHistory = document.getElementById('chat-history');

        // Hàm regex xử lý mọi loại ngoặc [cite_1], [[cite_1]], 【cite_1】
        function replaceCitations(text) {
            return text.replace(/[\[【]+cite_(\d+)[\]】]+/g, (match, id) => {
                return `<span class="citation-tag" onclick="jumpToCitation('${id}')">[cite_${id}]</span>`;
            });
        }

        function appendMessage(sender, text, isMarkdown = false, meta = null) {
            const div = document.createElement('div');
            div.className = "flex gap-4 fade-in mb-4";
            
            const isUser = sender === 'User';
            const iconBg = isUser ? 'bg-indigo-100 text-indigo-700' : 'bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-md';
            const bubbleBg = isUser ? 'bg-indigo-50 border-indigo-100 rounded-tr-none' : 'bg-white border-gray-100 rounded-tl-none';
            
            let contentHTML = isMarkdown ? marked.parse(text) : text;
            
            if (!isUser) {
                contentHTML = replaceCitations(contentHTML);
            }

            let metaHTML = '';
            if (meta) {
                let statusColor = "bg-gray-100 text-gray-700";
                if (meta.fact_check === 'FactCheckStatus.PASSED') statusColor = "bg-green-100 text-green-700 border-green-200";
                if (meta.fact_check === 'FactCheckStatus.FAILED') statusColor = "bg-red-100 text-red-700 border-red-200";
                
                metaHTML = `
                <div class="mt-3 pt-3 border-t border-gray-100 flex gap-2 text-xs flex-wrap">
                    <span class="px-2 py-1 rounded bg-gray-100 text-gray-600 font-mono">Intent: ${meta.intent}</span>
                    <span class="px-2 py-1 rounded border ${statusColor} font-mono">Fact-Check: ${meta.fact_check || 'N/A'}</span>
                </div>`;
            }

            div.innerHTML = `
                <div class="w-8 h-8 rounded-full ${iconBg} flex items-center justify-center font-bold shrink-0 text-sm">
                    ${isUser ? 'U' : 'AI'}
                </div>
                <div class="p-4 rounded-2xl shadow-sm border max-w-[85%] text-sm ${bubbleBg} text-gray-800">
                    <div class="markdown-body leading-relaxed content-area">${contentHTML}</div>
                    <div class="meta-area">${metaHTML}</div>
                </div>
            `;
            chatHistory.appendChild(div);
            chatHistory.scrollTop = chatHistory.scrollHeight;
            return div;
        }

        window.jumpToCitation = function(id) {
            const cite = currentCitations.find(c => c.citation_id === `cite_${id}`);
            if (!cite) return;
            
            pageNum = cite.page;
            queueRenderPage(pageNum);
            
            setTimeout(() => {
                bboxLayer.innerHTML = '';
                if (cite.bbox && cite.bbox.length === 4) {
                    const box = document.createElement('div');
                    box.className = 'bbox-highlight';
                    
                    // Lấy tọa độ chuẩn hóa từ API: [xmin, ymin, xmax, ymax] (từ 0.0 -> 1.0)
                    const [xmin, ymin, xmax, ymax] = cite.bbox;
                    
                    // Tính toán ra % so với kích thước viewport của PDF.js
                    box.style.top = (ymin * 100) + '%';
                    box.style.left = (xmin * 100) + '%';
                    box.style.height = ((ymax - ymin) * 100) + '%';
                    box.style.width = ((xmax - xmin) * 100) + '%';
                    
                    box.animate([
                        { opacity: 0.2 },
                        { opacity: 0.8 },
                        { opacity: 0.2 }
                    ], { duration: 1000, iterations: 3 });

                    bboxLayer.appendChild(box);
                    
                    // Cuộn chuột tới vị trí của BBox
                    const container = document.getElementById('pdf-container');
                    const scrollToY = (ymin * bboxLayer.clientHeight) - (container.clientHeight / 2);
                    container.scrollTo({ top: Math.max(0, scrollToY), behavior: 'smooth' });
                }
            }, 600); // Đợi PDF render xong
        };

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = queryInput.value.trim();
            if (!query) return;

            appendMessage('User', query);
            queryInput.value = '';
            
            const btn = document.getElementById('send-btn');
            btn.disabled = true;
            btn.innerHTML = `Đang nghĩ...`;

            const aiMsgDiv = appendMessage('AI', '...');
            const aiContent = aiMsgDiv.querySelector('.content-area');
            const aiMeta = aiMsgDiv.querySelector('.meta-area');
            
            try {
                const response = await fetch(API_BASE_URL + '/api/v1/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: query, 
                        company: CURRENT_COMPANY, 
                        year: CURRENT_YEAR 
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let answerText = '';
                
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6).replace(/\r$/, ''); // Chỉ xoá \r để giữ lại dấu cách của token
                            if (dataStr === '[DONE]') break;
                            
                            try {
                                const meta = JSON.parse(dataStr);
                                if (meta.citations) {
                                    currentCitations = meta.citations;
                                    let statusColor = "bg-gray-100 text-gray-700";
                                    if (meta.fact_check === 'FactCheckStatus.PASSED') statusColor = "bg-green-100 text-green-700 border-green-200";
                                    if (meta.fact_check === 'FactCheckStatus.FAILED') statusColor = "bg-red-100 text-red-700 border-red-200";
                                    aiMeta.innerHTML = `
                                    <div class="mt-3 pt-3 border-t border-gray-100 flex gap-2 text-xs flex-wrap">
                                        <span class="px-2 py-1 rounded bg-gray-100 text-gray-600 font-mono">Intent: ${meta.intent}</span>
                                        <span class="px-2 py-1 rounded border ${statusColor} font-mono">Fact-Check: ${meta.fact_check || 'N/A'}</span>
                                    </div>`;
                                }
                            } catch(e) {
                                if (answerText === '...') answerText = '';
                                answerText += dataStr;
                            }
                            
                            // Luôn luôn parse Markdown và replace Citations sau mỗi token
                            let contentHTML = marked.parse(answerText);
                            contentHTML = replaceCitations(contentHTML);
                            aiContent.innerHTML = contentHTML;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    }
                }
            } catch (err) {
                console.error(err);
                aiContent.innerHTML = `<span class="text-red-500">Lỗi kết nối tới máy chủ AI.</span>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `Gửi`;
            }
        });

        // Khởi động
        loadDocuments();
    