// Media Agent：唯一 AgentEvent 流的 Web 适配器。
(function () {
    'use strict';

    const page = document.querySelector('.agent-page');
    if (!page) return;

    const consoleNode = page.querySelector('.agent-console');
    const transcript = document.getElementById('agentTranscript');
    const composer = document.getElementById('agentComposer');
    const promptInput = document.getElementById('agentPrompt');
    const sendButton = document.getElementById('agentSend');
    const stopButton = document.getElementById('agentStop');
    const newSessionButton = document.getElementById('agentNewSession');
    const resumeButton = document.getElementById('agentResumeLatestSession');
    const historyButton = document.getElementById('toggleAgentRail');
    const historyRail = document.getElementById('agentHistoryRail');
    const sessionList = document.getElementById('agentSessionList');
    const sessionCount = document.getElementById('agentSessionCount');
    const sessionStatus = document.getElementById('agentSessionStatus');
    const responseStatus = document.getElementById('agentResponseStatus');
    const nextActions = document.getElementById('agentStartActions');
    const resumeSlot = document.getElementById('agentStartResume');
    const composerActions = composer?.querySelector('.agent-composer-actions');
    const nextActionsStatus = document.getElementById('agentStartActionsStatus');
    const newRepliesButton = document.getElementById('agentNewReplies');
    const sessionSearch = document.getElementById('agentSessionSearch');
    const DRAFT_PREFIX = 'mediaflux.agent.drafts.v1.';
    const DRAFT_TTL_MS = 6 * 60 * 60 * 1000;
    const MAX_DRAFTS = 20;

    const SESSION_KEY = 'mediaflux.agent.kernel.session.v1';
    const SESSION_RE = /^[A-Za-z0-9_-]{16,64}$/;
    const MAX_TRANSCRIPT_ITEMS = 120;
    const STREAM_MARKDOWN_INTERVAL_MS = 72;
    const MAX_MARKDOWN_DEPTH = 4;
    const TOOL_LABELS = {
        cloud: '读取光鸭云盘',
        guangya: '读取光鸭云盘',
        library: '查询媒体库',
        provider: '查询实时服务',
        downloads: '查询下载任务',
        download: '处理下载任务',
        indexer: '搜索资源',
        resource: '搜索资源',
        rss: '检查 RSS',
        media: '检查媒体订阅',
        discovery: '检索媒体信息',
        web: '查询公开信息',
        strm: '检查 STRM',
        local_media: '检查本地媒体',
        automation: '检查自动化任务',
        config: '检查项目配置',
    };

    let draftScope = '';
    let sessionId = storedSessionId() || createId('session');
    let sessionItems = [];
    let followOutput = true;
    let candidateExpiryTimer = null;
    const memoryDrafts = new Map();
    const sessionEdits = new Set();
    let latestSessionId = '';
    let activeRequest = null;
    let historyController = null;
    let sessionLoadGeneration = 0;
    let busy = false;

    function createId(prefix) {
        let value = '';
        if (globalThis.crypto?.randomUUID) {
            value = globalThis.crypto.randomUUID().replaceAll('-', '');
        } else if (globalThis.crypto?.getRandomValues) {
            const bytes = new Uint8Array(24);
            globalThis.crypto.getRandomValues(bytes);
            value = Array.from(bytes, (item) => item.toString(16).padStart(2, '0')).join('');
        } else {
            value = `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
        }
        return `${prefix}_${value}`.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64);
    }

    function storedSessionId() {
        try {
            const key = draftScope ? `${SESSION_KEY}.${draftScope}` : SESSION_KEY;
            const value = localStorage.getItem(key) || '';
            return SESSION_RE.test(value) ? value : '';
        } catch (_) { return ''; }
    }

    function rememberSession(value) {
        sessionId = value;
        try {
            localStorage.setItem(SESSION_KEY, value);
            if (draftScope) localStorage.setItem(`${SESSION_KEY}.${draftScope}`, value);
        } catch (_) { /* private mode */ }
    }

    function clipText(value, limit) {
        const text = String(value || '').slice(0, limit);
        // DOM maxlength 按 UTF-16 计数；截断时不要把 emoji 的代理对切成非法 JSON 文本。
        return /[\uD800-\uDBFF]$/.test(text) ? text.slice(0, -1) : text;
    }

    function readDrafts() {
        if (!draftScope) return {};
        try {
            const value = JSON.parse(sessionStorage.getItem(DRAFT_PREFIX + draftScope) || '{}');
            if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
            return Object.fromEntries(Object.entries(value).filter(([id, draft]) =>
                SESSION_RE.test(id) && typeof draft?.text === 'string' && draft.text.length <= 1000 &&
                Number.isFinite(draft.updated_at) && draft.updated_at <= Date.now() &&
                Date.now() - draft.updated_at < DRAFT_TTL_MS
            ).sort((a, b) => b[1].updated_at - a[1].updated_at).slice(0, MAX_DRAFTS));
        } catch (_) { return {}; }
    }

    function saveDraft() {
        const text = clipText(promptInput?.value, 1000);
        memoryDrafts.set(sessionId, text);
        if (memoryDrafts.size > MAX_DRAFTS) memoryDrafts.delete(memoryDrafts.keys().next().value);
        if (!draftScope) return;
        const drafts = readDrafts();
        const looksSensitive = /(?:password|passwd|api[_-]?key|secret|token|cookie|authorization|密码|密钥|令牌)\s*[:=]/i.test(text) || /-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(text);
        if (text && !looksSensitive) drafts[sessionId] = {text, updated_at: Date.now()};
        else delete drafts[sessionId];
        const bounded = Object.fromEntries(Object.entries(drafts)
            .sort((a, b) => b[1].updated_at - a[1].updated_at).slice(0, MAX_DRAFTS));
        try { sessionStorage.setItem(DRAFT_PREFIX + draftScope, JSON.stringify(bounded)); } catch (_) { /* storage optional */ }
        rememberSession(sessionId);
    }

    function removeDraft(id) {
        memoryDrafts.delete(id);
        if (!draftScope) return;
        const drafts = readDrafts();
        delete drafts[id];
        try { sessionStorage.setItem(DRAFT_PREFIX + draftScope, JSON.stringify(drafts)); } catch (_) { /* storage optional */ }
    }

    function restoreDraft() {
        if (!promptInput) return;
        promptInput.value = memoryDrafts.has(sessionId)
            ? memoryDrafts.get(sessionId) : (readDrafts()[sessionId]?.text || '');
        resizePrompt();
    }

    function configureDraftScope(value) {
        if (typeof value !== 'string' || !/^[a-f0-9]{32,64}$/.test(value) || draftScope === value) return;
        // 首次鉴权响应前输入的内容属于当前页面，不被迟到的持久草稿覆盖。
        let typed = String(promptInput?.value || '');
        const accountChanged = Boolean(draftScope);
        if (accountChanged) {
            // 同一页面的登录主体改变时，不把旧主体的内存/输入传给新主体。
            ++sessionLoadGeneration;
            activeRequest?.controller.abort();
            expireCandidateCards();
            memoryDrafts.clear();
            typed = '';
            if (promptInput) promptInput.value = '';
            transcript?.replaceChildren();
            followOutput = true;
            if (newRepliesButton) newRepliesButton.hidden = true;
            setConsoleEmpty(true);
        }
        draftScope = value;
        if (accountChanged) sessionId = storedSessionId() || createId('session');
        const scopedSession = storedSessionId();
        if (!busy && !typed && scopedSession) sessionId = scopedSession;
        if (!typed && !busy) restoreDraft();
        saveDraft();
    }

    function fillDraft(text) {
        const value = clipText(String(text || '').trim(), 1000);
        if (!promptInput || !value) return;
        if (promptInput.value.trim() && promptInput.value.trim() !== value) {
            announce(responseStatus, '输入框已有草稿，请先发送或清空后再选择。');
            window.showToast?.('已保留输入框中的草稿，请先发送或清空后再选择', 'warning');
            promptInput.focus();
            return;
        }
        promptInput.value = value;
        saveDraft();
        resizePrompt();
        promptInput.focus();
    }

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function icon(name) {
        const node = document.createElement('i');
        node.setAttribute('data-lucide', name);
        node.setAttribute('aria-hidden', 'true');
        return node;
    }

    function renderIcons(root) {
        window.renderLucideIcons?.(root || page);
    }

    function announce(node, value) {
        if (node) node.textContent = String(value || '');
    }

    function setConsoleEmpty(empty) {
        consoleNode?.classList.toggle('is-empty', Boolean(empty));
        const resumeParent = empty && resumeSlot ? resumeSlot : composerActions;
        if (resumeButton && resumeParent && resumeButton.parentElement !== resumeParent) {
            const wasFocused = document.activeElement === resumeButton;
            resumeParent.append(resumeButton);
            if (wasFocused && !resumeButton.disabled) resumeButton.focus({preventScroll: true});
        }
        if (promptInput) {
            promptInput.placeholder = empty
                ? (promptInput.dataset.emptyPlaceholder || '询问 MediaFlux')
                : (promptInput.dataset.activePlaceholder || '继续描述或调整任务');
        }
    }

    function transcriptNearBottom() {
        return !transcript || transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 140;
    }

    function scrollToBottom(force = false) {
        if (!transcript) return;
        if (!force && !followOutput) {
            if (newRepliesButton) newRepliesButton.hidden = false;
            return;
        }
        if (force) followOutput = true;
        if (newRepliesButton) newRepliesButton.hidden = true;
        requestAnimationFrame(() => {
            if (followOutput) transcript.scrollTop = transcript.scrollHeight;
        });
    }

    function pruneTranscript() {
        if (!transcript) return;
        while (transcript.children.length > MAX_TRANSCRIPT_ITEMS) {
            transcript.firstElementChild?.remove();
        }
    }

    function appendMessage(role, {recovered = false} = {}) {
        const item = element('article', `agent-message agent-message-${role}`);
        if (recovered) item.classList.add('is-recovered');
        const mark = element('div', 'agent-message-mark');
        mark.append(icon(role === 'user' ? 'user-round' : 'bot'));
        const body = element('div', 'agent-message-body');
        item.append(mark, body);
        transcript?.append(item);
        pruneTranscript();
        setConsoleEmpty(false);
        renderIcons(item);
        scrollToBottom(true);
        return {item, body};
    }

    function appendUser(text, options = {}) {
        const view = appendMessage('user', options);
        view.body.append(element('p', '', text));
        return view;
    }

    function appendText(parent, value) {
        const text = String(value || '');
        if (!text) return;
        const previous = parent.lastChild;
        if (previous?.nodeType === 3) previous.nodeValue += text;
        else parent.append(document.createTextNode(text));
    }

    function safeMarkdownLink(rawHref) {
        const value = String(rawHref || '').trim();
        if (!value) return null;
        if (value.startsWith('#') || (value.startsWith('/') && !value.startsWith('//')) || value.startsWith('?')) {
            return {href: value, external: false};
        }
        try {
            const parsed = new URL(value, window.location.href);
            const protocol = parsed.protocol.toLowerCase();
            if (!['http:', 'https:', 'mailto:'].includes(protocol)) return null;
            return {
                href: parsed.href,
                external: protocol === 'mailto:' || parsed.origin !== window.location.origin,
            };
        } catch (_) {
            return null;
        }
    }

    function trimBareUrl(value) {
        let url = String(value || '');
        while (/[.,;:!?，。；：！？》】）}]$/.test(url)) url = url.slice(0, -1);
        return url;
    }

    function appendMarkdownLink(parent, label, rawHref, depth) {
        const target = safeMarkdownLink(rawHref);
        if (!target) {
            appendInlineMarkdown(parent, label, depth + 1);
            return;
        }
        const anchor = element('a', 'agent-md-link');
        anchor.href = target.href;
        if (target.external) {
            anchor.target = '_blank';
            anchor.rel = 'noopener noreferrer';
        }
        if (/^(?:https?:\/\/|mailto:)/i.test(label)) appendText(anchor, label);
        else appendInlineMarkdown(anchor, label, depth + 1);
        parent.append(anchor);
    }

    function markdownLinkAt(source, start) {
        const image = source.startsWith('![', start);
        const labelStart = start + (image ? 2 : 1);
        const labelEnd = source.indexOf('](', labelStart);
        if (labelEnd < 0) return null;
        let cursor = labelEnd + 2;
        let nesting = 0;
        let escaped = false;
        for (; cursor < source.length; cursor += 1) {
            const character = source[cursor];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (character === '\\') {
                escaped = true;
                continue;
            }
            if (character === '(') nesting += 1;
            if (character === ')' && nesting > 0) nesting -= 1;
            else if (character === ')' && nesting === 0) break;
        }
        if (cursor >= source.length) return null;
        const destination = source.slice(labelEnd + 2, cursor).trim();
        const match = destination.match(/^(?:<([^>]+)>|([^\s]+))(?:\s+["'].*["'])?$/);
        if (!match) return null;
        return {
            image,
            label: source.slice(labelStart, labelEnd),
            href: match[1] || match[2] || '',
            end: cursor + 1,
        };
    }

    function appendInlineMarkdown(parent, input, depth = 0) {
        const source = String(input || '');
        if (!source || depth > MAX_MARKDOWN_DEPTH) {
            appendText(parent, source);
            return;
        }
        let index = 0;
        while (index < source.length) {
            const escapedCharacter = source[index + 1] || '';
            if (source[index] === '\\' && '\\`*{}[]()#+-.!_|>~'.includes(escapedCharacter)) {
                appendText(parent, source[index + 1]);
                index += 2;
                continue;
            }

            if (source[index] === '`') {
                const marker = source.slice(index).match(/^`+/)?.[0] || '`';
                const end = source.indexOf(marker, index + marker.length);
                if (end >= 0) {
                    const code = element('code', 'agent-md-inline-code', source.slice(index + marker.length, end).replace(/^ | $/g, ''));
                    parent.append(code);
                    index = end + marker.length;
                    continue;
                }
            }

            if (source.startsWith('![', index) || source[index] === '[') {
                const link = markdownLinkAt(source, index);
                if (link) {
                    if (link.image) {
                        const alt = element('span', 'agent-md-image-alt');
                        alt.setAttribute('role', 'img');
                        alt.setAttribute('aria-label', link.label || '图片');
                        alt.append(icon('image'), element('span', '', link.label || '图片'));
                        parent.append(alt);
                    } else {
                        appendMarkdownLink(parent, link.label, link.href, depth);
                    }
                    index = link.end;
                    continue;
                }
            }

            const strongMarker = source.startsWith('**', index) ? '**' : source.startsWith('__', index) ? '__' : '';
            if (strongMarker) {
                const end = source.indexOf(strongMarker, index + 2);
                if (end > index + 2) {
                    const strong = document.createElement('strong');
                    appendInlineMarkdown(strong, source.slice(index + 2, end), depth + 1);
                    parent.append(strong);
                    index = end + 2;
                    continue;
                }
            }

            if (source.startsWith('~~', index)) {
                const end = source.indexOf('~~', index + 2);
                if (end > index + 2) {
                    const deleted = document.createElement('del');
                    appendInlineMarkdown(deleted, source.slice(index + 2, end), depth + 1);
                    parent.append(deleted);
                    index = end + 2;
                    continue;
                }
            }

            const emphasisMarker = source[index] === '*' ? '*' : source[index] === '_' ? '_' : '';
            if (emphasisMarker) {
                const previous = source[index - 1] || '';
                const next = source[index + 1] || '';
                const canOpen = next && !/\s/.test(next) && !(emphasisMarker === '_' && /[\p{L}\p{N}]/u.test(previous));
                const end = canOpen ? source.indexOf(emphasisMarker, index + 1) : -1;
                if (end > index + 1) {
                    const emphasis = document.createElement('em');
                    appendInlineMarkdown(emphasis, source.slice(index + 1, end), depth + 1);
                    parent.append(emphasis);
                    index = end + 1;
                    continue;
                }
            }

            if (source[index] === '<') {
                const autoLink = source.slice(index).match(/^<(https?:\/\/[^>]+|mailto:[^>]+)>/i);
                if (autoLink) {
                    appendMarkdownLink(parent, autoLink[1], autoLink[1], depth);
                    index += autoLink[0].length;
                    continue;
                }
            }

            const beginsBareLink = source.startsWith('http://', index) || source.startsWith('https://', index);
            const bareLink = beginsBareLink ? source.slice(index).match(/^https?:\/\/[^\s<]+/i) : null;
            if (bareLink) {
                const href = trimBareUrl(bareLink[0]);
                appendMarkdownLink(parent, href, href, depth);
                index += href.length;
                continue;
            }

            let next = index + 1;
            while (next < source.length) {
                const character = source[next];
                if ('\\`*_[~<'.includes(character)
                    || character === '['
                    || (character === '!' && source[next + 1] === '[')
                    || source.startsWith('http://', next)
                    || source.startsWith('https://', next)) break;
                next += 1;
            }
            appendText(parent, source.slice(index, next));
            index = next;
        }
    }

    function splitMarkdownTableRow(value) {
        let line = String(value || '').trim();
        if (line.startsWith('|')) line = line.slice(1);
        if (line.endsWith('|')) line = line.slice(0, -1);
        const cells = [];
        let current = '';
        let codeMarker = false;
        for (let index = 0; index < line.length; index += 1) {
            const character = line[index];
            const next = line[index + 1] || '';
            if (character === '\\' && ['|', '\\', '`'].includes(next)) {
                current += next;
                index += 1;
                continue;
            }
            if (character === '`') codeMarker = !codeMarker;
            if (character === '|' && !codeMarker) {
                cells.push(current.trim());
                current = '';
            } else {
                current += character;
            }
        }
        cells.push(current.trim());
        return cells;
    }

    function markdownTableDefinition(lines, index) {
        if (index + 1 >= lines.length || !lines[index].includes('|')) return null;
        const header = splitMarkdownTableRow(lines[index]);
        const separators = splitMarkdownTableRow(lines[index + 1]);
        if (header.length < 2 || header.length !== separators.length) return null;
        if (!separators.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, '')))) return null;
        return {header, separators};
    }

    function markdownListItem(value) {
        const unordered = String(value || '').match(/^\s{0,3}[-+*•]\s+(.+)$/);
        if (unordered) return {ordered: false, value: unordered[1], start: 1};
        const ordered = String(value || '').match(/^\s{0,3}(\d{1,4})[.)、]\s+(.+)$/);
        return ordered ? {ordered: true, value: ordered[2], start: Number(ordered[1]) || 1} : null;
    }

    function isMarkdownBlockStart(lines, index) {
        const line = String(lines[index] || '');
        return /^\s{0,3}(?:`{3,}|~{3,})/.test(line)
            || /^\s{0,3}#{1,6}\s+/.test(line)
            || /^\s{0,3}>/.test(line)
            || /^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(line)
            || Boolean(markdownListItem(line))
            || Boolean(markdownTableDefinition(lines, index));
    }

    function appendMarkdownBlocks(root, input, depth = 0) {
        const lines = String(input || '').replace(/\r\n?/g, '\n').split('\n');
        let index = 0;
        while (index < lines.length) {
            const raw = lines[index];
            const line = raw.trim();
            if (!line) {
                index += 1;
                continue;
            }

            const fence = raw.match(/^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)\s*$/);
            if (fence) {
                const marker = fence[1];
                const language = String(fence[2] || '').replace(/[^A-Za-z0-9_+.-]/g, '').slice(0, 24);
                const values = [];
                index += 1;
                while (index < lines.length && !new RegExp(`^\\s{0,3}${marker[0]}{${marker.length},}\\s*$`).test(lines[index])) {
                    values.push(lines[index]);
                    index += 1;
                }
                if (index < lines.length) index += 1;
                const frame = element('div', 'agent-md-code-frame');
                if (language) frame.append(element('span', 'agent-md-code-language', language));
                const pre = document.createElement('pre');
                pre.append(element('code', '', values.join('\n')));
                frame.append(pre);
                root.append(frame);
                continue;
            }

            const heading = raw.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
            if (heading) {
                const level = heading[1].length;
                const title = element(`h${Math.min(6, level + 1)}`, `agent-md-heading agent-md-heading-${level}`);
                appendInlineMarkdown(title, heading[2]);
                root.append(title);
                index += 1;
                continue;
            }

            if (/^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(raw)) {
                root.append(document.createElement('hr'));
                index += 1;
                continue;
            }

            if (/^\s{0,3}>/.test(raw)) {
                const quoteLines = [];
                while (index < lines.length && /^\s{0,3}>/.test(lines[index])) {
                    quoteLines.push(lines[index].replace(/^\s{0,3}>\s?/, ''));
                    index += 1;
                }
                const quote = document.createElement('blockquote');
                if (depth < MAX_MARKDOWN_DEPTH) appendMarkdownBlocks(quote, quoteLines.join('\n'), depth + 1);
                else appendText(quote, quoteLines.join('\n'));
                root.append(quote);
                continue;
            }

            const table = markdownTableDefinition(lines, index);
            if (table) {
                const scroll = element('div', 'agent-md-table-scroll');
                scroll.tabIndex = 0;
                scroll.setAttribute('role', 'region');
                scroll.setAttribute('aria-label', 'Markdown 表格');
                const tableNode = document.createElement('table');
                const head = document.createElement('thead');
                const headRow = document.createElement('tr');
                table.header.forEach((value, cellIndex) => {
                    const cell = document.createElement('th');
                    const separator = table.separators[cellIndex].replace(/\s/g, '');
                    if (separator.startsWith(':') && separator.endsWith(':')) cell.className = 'is-center';
                    else if (separator.endsWith(':')) cell.className = 'is-right';
                    appendInlineMarkdown(cell, value);
                    headRow.append(cell);
                });
                head.append(headRow);
                tableNode.append(head);
                const body = document.createElement('tbody');
                index += 2;
                while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
                    const values = splitMarkdownTableRow(lines[index]);
                    const row = document.createElement('tr');
                    table.header.forEach((_, cellIndex) => {
                        const cell = document.createElement('td');
                        const separator = table.separators[cellIndex].replace(/\s/g, '');
                        if (separator.startsWith(':') && separator.endsWith(':')) cell.className = 'is-center';
                        else if (separator.endsWith(':')) cell.className = 'is-right';
                        appendInlineMarkdown(cell, values[cellIndex] || '');
                        row.append(cell);
                    });
                    body.append(row);
                    index += 1;
                }
                tableNode.append(body);
                scroll.append(tableNode);
                root.append(scroll);
                continue;
            }

            const listItem = markdownListItem(raw);
            if (listItem) {
                const list = document.createElement(listItem.ordered ? 'ol' : 'ul');
                if (listItem.ordered && listItem.start !== 1) list.start = listItem.start;
                while (index < lines.length) {
                    const item = markdownListItem(lines[index]);
                    if (!item || item.ordered !== listItem.ordered) break;
                    const row = document.createElement('li');
                    const task = item.value.match(/^\[([ xX])\]\s+(.+)$/);
                    if (task) {
                        row.className = 'agent-md-task';
                        const marker = element('span', 'agent-md-task-marker', task[1].toLowerCase() === 'x' ? '✓' : '');
                        marker.setAttribute('aria-hidden', 'true');
                        row.append(marker);
                        appendInlineMarkdown(row, task[2]);
                    } else {
                        appendInlineMarkdown(row, item.value);
                    }
                    list.append(row);
                    index += 1;
                }
                root.append(list);
                continue;
            }

            const paragraphLines = [];
            while (index < lines.length && lines[index].trim() && (paragraphLines.length === 0 || !isMarkdownBlockStart(lines, index))) {
                paragraphLines.push(lines[index]);
                index += 1;
            }
            const paragraphText = paragraphLines.map((value) => value.trim()).join(' ');
            const paragraph = element('p', root.childElementCount === 0 && paragraphText.length <= 140 ? 'agent-answer-lead' : '');
            paragraphLines.forEach((value, lineIndex) => {
                const hardBreak = /\s{2}$/.test(value) || /\\$/.test(value);
                appendInlineMarkdown(paragraph, value.replace(/(?:\s{2}|\\)$/, '').trim());
                if (lineIndex < paragraphLines.length - 1) paragraph.append(hardBreak ? document.createElement('br') : document.createTextNode(' '));
            });
            root.append(paragraph);
        }
    }

    function parseTextBlocks(text) {
        const root = element('div', 'agent-rich-text');
        appendMarkdownBlocks(root, text);
        return root;
    }

    function replaceRichText(target, text) {
        const rendered = parseTextBlocks(text);
        target.classList.add('agent-rich-text');
        target.replaceChildren(...rendered.childNodes);
        if (target.querySelector('[data-lucide]')) renderIcons(target);
    }

    function cancelTurnMarkdownRender(turn) {
        if (!turn) return;
        if (turn.markdownTimer !== null) window.clearTimeout(turn.markdownTimer);
        if (turn.markdownFrame !== null) window.cancelAnimationFrame(turn.markdownFrame);
        turn.markdownTimer = null;
        turn.markdownFrame = null;
    }

    function renderTurnMarkdown(turn, text, {immediate = false} = {}) {
        if (!turn?.text) return;
        turn.pendingMarkdown = String(text || '');
        const commit = () => {
            const shouldFollow = transcriptNearBottom();
            turn.markdownTimer = null;
            turn.markdownFrame = null;
            replaceRichText(turn.text, turn.pendingMarkdown);
            turn.lastMarkdownRender = performance.now();
            if (shouldFollow) scrollToBottom(true);
        };
        if (immediate) {
            cancelTurnMarkdownRender(turn);
            commit();
            return;
        }
        if (turn.markdownTimer !== null || turn.markdownFrame !== null) return;
        const delay = Math.max(0, STREAM_MARKDOWN_INTERVAL_MS - (performance.now() - turn.lastMarkdownRender));
        turn.markdownTimer = window.setTimeout(() => {
            turn.markdownTimer = null;
            turn.markdownFrame = window.requestAnimationFrame(commit);
        }, delay);
    }

    function createAssistantTurn({recovered = false} = {}) {
        const view = appendMessage('assistant', {recovered});
        const card = element('section', 'agent-result-card agent-streaming');
        const head = element('div', 'agent-stream-head');
        head.append(icon('loader-circle'), element('span', '', '正在理解任务'));
        const text = element('div', 'agent-stream-text agent-rich-text');
        const steps = element('div', 'agent-stream-steps');
        card.append(head, text, steps);
        view.body.append(card);
        renderIcons(card);
        return {
            ...view,
            card,
            head,
            headText: head.querySelector('span'),
            text,
            steps,
            rounds: new Map(),
            currentRound: 0,
            toolSteps: new Map(),
            approvalNode: null,
            pendingMarkdown: '',
            markdownTimer: null,
            markdownFrame: null,
            lastMarkdownRender: 0,
        };
    }

    function setTurnStatus(turn, label, iconName = 'loader-circle') {
        if (!turn?.head) return;
        turn.head.replaceChildren(icon(iconName), element('span', '', label));
        renderIcons(turn.head);
    }

    function toolLabel(tool, label = '') {
        const explicit = String(label || '').trim();
        if (explicit) return explicit;
        const prefix = String(tool || '').split('.', 1)[0].toLowerCase();
        return TOOL_LABELS[prefix] || '调用项目能力';
    }

    function updateStep(turn, key, label, {warning = false, pending = false} = {}) {
        if (!turn?.steps || !key) return;
        let row = turn.toolSteps.get(key);
        if (!row) {
            row = element('div', 'agent-stream-step');
            row.dataset.stepKey = key;
            turn.toolSteps.set(key, row);
            turn.steps.append(row);
        }
        row.classList.toggle('is-warning', warning);
        row.classList.toggle('is-pending', pending);
        row.replaceChildren(
            icon(pending ? 'loader-circle' : warning ? 'triangle-alert' : 'check'),
            element('span', '', label),
        );
        renderIcons(row);
        scrollToBottom();
    }

    function buildToolTrace(turn) {
        if (!turn?.steps || !turn.steps.childElementCount) {
            turn?.steps?.remove();
            return null;
        }
        const trace = element('details', 'agent-tool-trace');
        const summary = element('summary', 'agent-tool-trace-summary');
        summary.append(
            icon('list-checks'),
            element('span', '', `执行过程 · ${turn.steps.childElementCount} 步`),
            icon('chevron-down'),
        );
        trace.append(summary, turn.steps);
        renderIcons(trace);
        return trace;
    }

    function addRecoveredToolTrace(turn, tools, labels = []) {
        if (!Array.isArray(tools)) return;
        for (const [index, name] of tools.entries()) {
            const normalized = String(name || '').trim();
            if (!normalized) continue;
            updateStep(turn, `recovered:${index}:${normalized}`, `${toolLabel(normalized, labels[index])}完成`);
        }
    }

    function publicSummary(value) {
        if (value && typeof value === 'object') {
            for (const key of ['summary', 'message', 'title', 'status']) {
                if (typeof value[key] === 'string' && value[key].trim()) return value[key].trim();
            }
        }
        return typeof value === 'string' ? value.trim() : '';
    }

    function finalizeAnswer(turn, text) {
        turn.failed = false;
        cancelTurnMarkdownRender(turn);
        const answer = String(text || '').trim();
        if (!answer && turn.candidateGroup) {
            turn.head?.remove();
            turn.text?.remove();
            scrollToBottom();
            return;
        }
        if (!answer) {
            turn.item?.remove();
            setConsoleEmpty(!transcript?.childElementCount);
            return;
        }
        turn.card.classList.remove('agent-streaming', 'is-interrupted');
        turn.card.classList.add('has-narrative', 'is-conversation');
        turn.head.remove();
        turn.text.className = 'agent-narrative agent-rich-text';
        replaceRichText(turn.text, answer);
        const trace = buildToolTrace(turn);
        if (trace) turn.card.append(trace);
        scrollToBottom();
    }

    function finalizeError(turn, message, {cancelled = false} = {}) {
        turn.failed = true;
        turn.cancelled = cancelled;
        cancelTurnMarkdownRender(turn);
        turn.card.classList.remove('agent-streaming');
        turn.card.classList.add(cancelled ? 'agent-cancelled' : 'is-interrupted');
        setTurnStatus(turn, cancelled ? '已停止' : '未能完成', cancelled ? 'circle-stop' : 'triangle-alert');
        turn.text.textContent = message || (cancelled ? '本次任务已停止。' : 'Agent 暂时无法完成该请求。');
        const trace = buildToolTrace(turn);
        if (trace) turn.card.append(trace);
        if (turn.requestMessage && !turn.boundSelection && !turn.card.querySelector('.agent-retry-draft')) {
            const actions = element('div', 'agent-retry-actions');
            const retry = element('button', 'agent-retry-draft', '放回输入框修改');
            retry.type = 'button';
            retry.dataset.agentDraft = turn.requestMessage;
            actions.append(retry);
            turn.card.append(actions);
        }
        scrollToBottom();
    }

    function approvalTargetLabel(value) {
        const target = String(value || '').trim().toLowerCase();
        return ({guangya: '光鸭云盘', qb: 'qBittorrent', qbittorrent: 'qBittorrent'})[target] || target;
    }

    function scalarPreviewRows(data, confirmation = {}) {
        if (!data || typeof data !== 'object' || Array.isArray(data)) return [];
        const rows = [];
        const object = String(confirmation.object || '').trim();
        if (object) rows.push(['操作对象', object.slice(0, 320)]);
        const target = approvalTargetLabel(data.target);
        if (target) rows.push(['目标', target.slice(0, 80)]);
        const count = Number.isInteger(data.count) ? data.count : Number.isInteger(data.total) ? data.total : null;
        if (count !== null) rows.push(['数量', `${count} 项`]);
        for (const [key, label] of [['selected', '已选择'], ['review_required', '待复核']]) {
            if (Number.isInteger(data[key])) rows.push([label, `${data[key]} 项`]);
        }
        return rows.slice(0, 6);
    }

    function approvalListItem(value) {
        if (typeof value === 'string') return value.trim();
        if (!value || typeof value !== 'object') return '';
        for (const key of ['title', 'summary', 'action', 'description', 'name']) {
            if (typeof value[key] === 'string' && value[key].trim()) return value[key].trim();
        }
        return '';
    }

    function buildApprovalScope(data) {
        if (!data || typeof data !== 'object') return null;
        const resources = Array.isArray(data.resources) ? data.resources : [];
        const effects = Array.isArray(data.effects) ? data.effects : [];
        if (!resources.length && !effects.length) return null;
        const scope = element('div', 'agent-confirmation-scope');
        if (resources.length) {
            scope.append(element('h4', '', '将处理'));
            const list = element('ul', 'agent-confirmation-list');
            for (const item of resources.slice(0, 8)) {
                if (!item || typeof item !== 'object') continue;
                const title = approvalListItem(item) || '未命名资源';
                const site = String(item.site_name || '').trim();
                const position = Number.isInteger(item.position) ? `#${item.position} · ` : '';
                list.append(element('li', '', `${position}${title}${site ? ` · ${site}` : ''}`));
            }
            if (resources.length > 8) list.append(element('li', 'is-muted', `另有 ${resources.length - 8} 项`));
            if (list.childElementCount) scope.append(list);
        }
        if (effects.length) {
            scope.append(element('h4', '', '执行内容'));
            const list = element('ul', 'agent-confirmation-list');
            for (const item of effects.slice(0, 5)) {
                const text = approvalListItem(item);
                if (text) list.append(element('li', '', text));
            }
            if (list.childElementCount) scope.append(list);
        }
        return scope.childElementCount ? scope : null;
    }

    const unconfirmedEffectMessage = '执行结果尚未确认，请先查询实际业务状态，勿直接重复提交。';

    function hasEffectResult(result) {
        return result && typeof result === 'object' && !Array.isArray(result) && Object.keys(result).length > 0;
    }

    function formatEffectResult(result) {
        if (!hasEffectResult(result)) return `⚠️ ${unconfirmedEffectMessage}`;
        const summary = publicSummary(result) || '操作已结束。';
        const status = String(result.status || '').toLowerCase();
        const iconPrefix = result.ok === false || ['failed', 'error'].includes(status)
            ? '❌'
            : ['partial', 'degraded', 'incomplete', 'attention'].includes(status) ? '⚠️' : '✅';
        const lines = [`${iconPrefix} ${summary}`];
        const data = result.data;
        if (data && typeof data === 'object' && !Array.isArray(data)) {
            const target = approvalTargetLabel(data.target);
            if (target) lines.push(`- 目标：${target}`);
            for (const [key, label] of [['total', '请求'], ['succeeded', '已受理'], ['created', '已创建'], ['review_required', '待复核'], ['duplicate', '已存在'], ['failed', '未完成'], ['skipped', '已跳过']]) {
                if (Number.isInteger(data[key])) lines.push(`- ${label}：${data[key]} 项`);
            }
            if (Array.isArray(data.items)) {
                const errors = [];
                for (const item of data.items) {
                    const value = item && item.ok === false ? String(item.error || '').trim() : '';
                    if (value && !errors.includes(value)) errors.push(value);
                    if (errors.length >= 3) break;
                }
                for (const value of errors) lines.push(`- 失败原因：${value}`);
            }
        }
        if (typeof result.error === 'string' && result.error.trim() && !lines.join('\n').includes(result.error.trim())) {
            lines.push(`- 说明：${result.error.trim()}`);
        }
        return lines.join('\n');
    }

    function buildApproval(approval) {
        const card = element('section', 'agent-confirmation-card');
        if (String(approval.effect || '').toUpperCase() === 'DANGER') {
            card.classList.add('is-risk-danger');
        }
        card.dataset.planId = approval.plan_id || '';
        const head = element('div', 'agent-confirmation-head');
        const heading = element('div', 'agent-confirmation-heading');
        const title = element('div', 'agent-confirmation-title');
        const confirmation = approval.confirmation && typeof approval.confirmation === 'object' ? approval.confirmation : {};
        title.append(
            element('span', '', '安全执行计划'),
            element('strong', '', String(confirmation.action || '确认后执行变更')),
        );
        heading.append(title);
        const risk = element('span', 'agent-confirmation-risk', String(approval.effect || 'WRITE').toUpperCase() === 'DANGER' ? '高风险' : '需确认');
        if (String(approval.effect || '').toUpperCase() === 'DANGER') risk.classList.add('is-danger');
        head.append(heading, risk);

        const intro = element('div', 'agent-confirmation-intro');
        intro.append(parseTextBlocks(String(confirmation.preflight_summary || '').trim() || publicSummary(approval.preview) || publicSummary(approval.result) || '预检已完成。'));
        const facts = element('dl', 'agent-confirmation-facts');
        const previewData = approval.preview?.data;
        for (const [key, value] of scalarPreviewRows(previewData, confirmation)) {
            const row = element('div', 'agent-confirmation-fact');
            row.append(element('dt', '', key), element('dd', '', value));
            facts.append(row);
        }
        const scope = buildApprovalScope(previewData);
        const details = element('dl', 'agent-confirmation-details');
        for (const [label, value] of [['执行影响', confirmation.impact], ['如何撤销', confirmation.reversibility]]) {
            const text = String(value || '').trim();
            if (!text) continue;
            const row = element('div', 'agent-confirmation-detail');
            row.append(element('dt', '', label), element('dd', '', text));
            details.append(row);
        }
        const status = element('div', 'agent-confirmation-status');
        const preflight = element('p', 'agent-confirmation-preflight');
        preflight.append(icon('shield-check'), element('span', '', '系统只冻结了计划，尚未写入任何变更。'));
        status.append(preflight);
        if (approval.expires_at) {
            const expiry = element('p', 'agent-confirmation-copy');
            expiry.append(icon('clock-3'), element('span', 'agent-confirmation-time-copy', `有效期至 ${approval.expires_at}`));
            status.append(expiry);
        }
        const actions = element('div', 'agent-confirmation-actions');
        const cancel = element('button', 'agent-confirmation-cancel', '取消');
        cancel.type = 'button';
        cancel.dataset.effectCancel = approval.plan_id || '';
        const confirm = element('button', 'agent-confirmation-submit', '确认执行');
        confirm.type = 'button';
        confirm.dataset.effectConfirm = approval.plan_id || '';
        actions.append(cancel, confirm);
        card.append(head, intro);
        if (facts.childElementCount) card.append(facts);
        if (scope) card.append(scope);
        if (details.childElementCount) card.append(details);
        card.append(status, actions);
        renderIcons(card);
        return card;
    }

    function showApproval(turn, approval) {
        const card = buildApproval(approval);
        const trace = buildToolTrace(turn);
        if (trace) {
            const status = card.querySelector('.agent-confirmation-status');
            card.insertBefore(trace, status || null);
        }
        // 确认卡包含真实写操作按钮，不应继承消息入场位移动画；否则在快速
        // 预检完成时按钮会短暂移动，既影响触控，也会造成自动化点击不稳定。
        turn.item?.classList.add('is-confirmation');
        turn.card.replaceWith(card);
        turn.approvalNode = card;
        scrollToBottom(true);
    }

    function replaceApprovalWithResult(card, text, {error = false, cancelled = false} = {}) {
        const result = element('section', `agent-result-card${error ? ' is-interrupted' : ''}${cancelled ? ' agent-cancelled' : ''}`);
        const head = element('div', 'agent-stream-head');
        head.append(icon(error ? 'triangle-alert' : cancelled ? 'circle-stop' : 'circle-check-big'), element('span', '', error ? '执行失败' : cancelled ? '已取消' : '执行完成'));
        const body = element('div', 'agent-stream-text agent-rich-text');
        replaceRichText(body, text);
        result.append(head, body);
        card.replaceWith(result);
        renderIcons(result);
        scrollToBottom(true);
    }

    function expireVisibleApprovals() {
        transcript?.querySelectorAll('.agent-confirmation-card[data-plan-id]').forEach((card) => {
            card.classList.add('is-expired');
            card.querySelectorAll('button').forEach((button) => { button.disabled = true; });
            const status = card.querySelector('.agent-confirmation-preflight span');
            if (status) status.textContent = '已由新的任务替代，本计划不会执行。';
        });
    }

    function applyEvent(turn, event) {
        const payload = event?.payload && typeof event.payload === 'object' ? event.payload : {};
        switch (event?.type) {
        case 'turn.started':
            if (turn.boundSelection) expireVisibleApprovals();
            setTurnStatus(turn, payload.kind === 'confirmation' ? '正在执行已确认计划' : '正在理解任务');
            break;
        case 'capabilities.selected':
            setTurnStatus(turn, '已准备相关能力，正在规划');
            break;
        case 'model.started':
            turn.currentRound = Number(payload.round || turn.currentRound + 1);
            turn.rounds.set(turn.currentRound, '');
            setTurnStatus(turn, turn.currentRound > 1 ? '正在汇总结果' : '正在规划下一步');
            break;
        case 'model.delta': {
            const round = Number(payload.round || turn.currentRound || 1);
            const value = `${turn.rounds.get(round) || ''}${String(payload.delta || '')}`;
            turn.rounds.set(round, value);
            renderTurnMarkdown(turn, value);
            break;
        }
        case 'model.tool_call': {
            const key = `call:${payload.call_id || event.sequence}`;
            updateStep(turn, key, `${toolLabel(payload.tool, payload.label)}…`, {pending: true});
            cancelTurnMarkdownRender(turn);
            turn.pendingMarkdown = '';
            turn.text.replaceChildren();
            setTurnStatus(turn, toolLabel(payload.tool, payload.label));
            break;
        }
        case 'tool.started':
            setTurnStatus(turn, toolLabel(payload.tool, payload.label));
            break;
        case 'tool.progress': {
            if (Object.prototype.hasOwnProperty.call(payload, 'candidate_view') && payload.candidate_view === null) expireCandidateCards();
            const summary = publicSummary(payload);
            if (summary) setTurnStatus(turn, summary.slice(0, 100));
            break;
        }
        case 'tool.completed':
            if (payload.result?.candidate_view) renderCandidateView(turn, payload.result.candidate_view);
            else if (Object.prototype.hasOwnProperty.call(payload.result || {}, 'candidate_view')) expireCandidateCards();
            updateStep(turn, `call:${payload.call_id || event.sequence}`, `${toolLabel(payload.tool, payload.label)}完成`);
            break;
        case 'tool.failed':
            updateStep(turn, `call:${payload.call_id || event.sequence}`, `${toolLabel(payload.tool, payload.label)}未完成，正在调整`, {warning: true});
            setTurnStatus(turn, '正在调整方案');
            break;
        case 'effect.preview_started':
            setTurnStatus(turn, '正在生成安全变更预览');
            break;
        case 'effect.approval_required':
            if (payload.plan) {
                updateStep(
                    turn,
                    `call:${payload.call_id || event.sequence}`,
                    `${toolLabel(payload.tool, payload.label)}预检完成`,
                );
                showApproval(turn, {
                    plan_id: payload.plan.plan_id,
                    tool_name: payload.plan.tool_name || payload.tool,
                    effect: payload.plan.effect,
                    preview: payload.plan.preview || {},
                    result: payload.result || {},
                    confirmation: payload.plan.confirmation || {},
                    expires_at: payload.plan.expires_at || '',
                });
            }
            break;
        case 'effect.completed':
            turn.effectResult = payload.result || {};
            break;
        case 'effect.failed':
            turn.effectError = hasEffectResult(payload.result)
                ? formatEffectResult({...payload.result, ok: false, error: payload.result.error || payload.message})
                : payload.message || '确认执行失败。';
            break;
        case 'turn.completed':
            if (payload.status === 'success') finalizeAnswer(turn, payload.answer || '');
            else if (payload.status === 'effect_completed') {
                finalizeAnswer(turn, formatEffectResult(turn.effectResult));
            }
            break;
        case 'turn.failed':
            finalizeError(turn, turn.effectError || payload.message || 'Agent 暂时无法完成该请求。');
            break;
        case 'turn.cancelled':
            finalizeError(turn, '本次任务已停止。', {cancelled: true});
            break;
        default:
            break;
        }
    }

    async function readEventStream(response, consume) {
        if (!response.ok) {
            const text = await response.text();
            let message = `请求失败（HTTP ${response.status}）`;
            try { message = JSON.parse(text).error || message; } catch (_) { /* non-json */ }
            throw new Error(message);
        }
        if (!response.body?.getReader) throw new Error('当前浏览器不支持流式响应');
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8', {fatal: true});
        let buffer = '';
        try {
            while (true) {
                const {value, done} = await reader.read();
                if (value) buffer += decoder.decode(value, {stream: !done});
                let lineEnd = buffer.indexOf('\n');
                while (lineEnd >= 0) {
                    const line = buffer.slice(0, lineEnd).trim();
                    buffer = buffer.slice(lineEnd + 1);
                    if (line) consume(JSON.parse(line));
                    lineEnd = buffer.indexOf('\n');
                }
                if (done) break;
            }
            buffer += decoder.decode();
            if (buffer.trim()) consume(JSON.parse(buffer.trim()));
        } finally {
            try { reader.releaseLock(); } catch (_) { /* already released */ }
        }
    }

    async function fetchJSON(url, options = {}) {
        const response = await fetch(url, options);
        const text = await response.text();
        let payload = {};
        if (text) {
            try { payload = JSON.parse(text); } catch (_) { payload = {}; }
        }
        if (!response.ok) throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
        return payload;
    }

    function setBusy(value, {stoppable = false} = {}) {
        busy = Boolean(value);
        if (promptInput) promptInput.disabled = false;
        if (sendButton) {
            sendButton.hidden = busy && stoppable;
            sendButton.disabled = busy || !promptInput?.value.trim();
            sendButton.setAttribute('aria-busy', String(busy));
        }
        if (stopButton) {
            stopButton.hidden = !(busy && stoppable);
            stopButton.disabled = !(busy && stoppable);
        }
        syncCandidateButtons();
        newSessionButton && (newSessionButton.disabled = busy);
        resumeButton && (resumeButton.disabled = busy || !latestSessionId);
    }

    function syncSend() {
        if (sendButton && !busy) sendButton.disabled = !promptInput?.value.trim();
    }

    function resizePrompt() {
        if (!promptInput) return;
        promptInput.style.height = 'auto';
        promptInput.style.height = `${Math.min(160, Math.max(44, promptInput.scrollHeight))}px`;
        syncSend();
        syncViewportHeight();
    }

    async function sendQuery(text, {selection = null, preserveDraft = false} = {}) {
        if (busy || !text.trim()) return;
        const message = text.trim();
        ++sessionLoadGeneration;
        expireCandidateCards();
        if (!selection) expireVisibleApprovals();
        appendUser(message);
        const turn = createAssistantTurn();
        turn.requestMessage = message;
        turn.boundSelection = Boolean(selection);
        if (!preserveDraft) promptInput.value = '';
        saveDraft();
        scrollToBottom(true);
        resizePrompt();
        rememberSession(sessionId);
        const controller = new AbortController();
        const requestId = createId('rq');
        activeRequest = {controller, requestId, turn, sessionId};
        setBusy(true, {stoppable: true});
        announce(responseStatus, 'Media Agent 正在处理请求');
        try {
            const response = await fetch('/api/agent/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message,
                    session_id: sessionId,
                    request_id: requestId,
                    stream: true,
                    ...(selection ? {selection} : {}),
                }),
                signal: controller.signal,
            });
            await readEventStream(response, (event) => {
                if (activeRequest?.requestId !== requestId) return;
                applyEvent(turn, event);
            });
            announce(responseStatus, turn.failed ? (turn.cancelled ? '请求已停止' : '请求失败') : 'Media Agent 已完成');
        } catch (error) {
            if (error?.name === 'AbortError') finalizeError(turn, '本次任务已停止。', {cancelled: true});
            else finalizeError(turn, error?.message || 'Agent 暂时不可用。');
            announce(responseStatus, error?.name === 'AbortError' ? '请求已停止' : '请求失败');
        } finally {
            if (activeRequest?.requestId === requestId) activeRequest = null;
            setBusy(false);
            resizePrompt();
            refreshSessions({quiet: true});
        }
    }

    async function stopActiveRequest() {
        const active = activeRequest;
        if (!active) return;
        stopButton.disabled = true;
        try {
            await Promise.race([
                fetchJSON('/api/agent/query/cancel', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: active.sessionId, request_id: active.requestId}),
                }),
                new Promise((resolve) => setTimeout(resolve, 1200)),
            ]);
        } catch (_) { /* stream abort remains authoritative for the browser */ }
        active.controller.abort();
    }

    async function confirmEffect(button) {
        if (busy) return;
        const card = button.closest('.agent-confirmation-card');
        const planId = button.dataset.effectConfirm || '';
        if (!card || !planId) return;
        const buttons = [...card.querySelectorAll('button')];
        buttons.forEach((item) => { item.disabled = true; });
        const actions = card.querySelector('.agent-confirmation-actions');
        const executing = element('div', 'agent-confirmation-executing');
        const mark = element('span', 'agent-confirmation-executing-mark');
        mark.append(icon('loader-circle'));
        const copy = element('span', 'agent-confirmation-executing-copy');
        copy.append(element('strong', '', '正在执行已确认计划'), element('small', '', '执行完成前不会接受另一项写操作。'));
        executing.append(mark, copy);
        actions?.replaceChildren(executing);
        renderIcons(card);
        setBusy(true);
        const controller = new AbortController();
        const requestId = createId('confirm');
        let result = null;
        let effectTerminal = '';
        let failure = '';
        let transportError = '';
        try {
            const response = await fetch('/api/agent/actions/confirm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    plan_id: planId,
                    session_id: sessionId,
                    request_id: requestId,
                    stream: true,
                }),
                signal: controller.signal,
            });
            await readEventStream(response, (event) => {
                const payload = event.payload || {};
                if (event.type === 'effect.completed') {
                    effectTerminal = 'completed';
                    result = payload.result;
                    failure = '';
                } else if (event.type === 'effect.failed') {
                    effectTerminal = 'failed';
                    result = payload.result;
                    failure = payload.message || '确认执行未能完成。';
                } else if (event.type === 'turn.failed' && effectTerminal !== 'failed') {
                    failure = payload.message || '确认执行未能完成。';
                } else if (event.type === 'turn.cancelled' && !effectTerminal) {
                    failure = payload.reason || unconfirmedEffectMessage;
                }
            });
        } catch (error) {
            transportError = error?.message || '事件流中断';
        } finally {
            // 只有可信的 effect 终态可确认写入；EOF/缺失 DTO 不能补成成功。
            // 已收到的业务终态优先于后续传输错误，失败 DTO 优先于笼统 turn.failed。
            const completed = effectTerminal === 'completed' && hasEffectResult(result);
            const failed = effectTerminal === 'failed' || Boolean(failure);
            let text = unconfirmedEffectMessage;
            if (effectTerminal === 'failed' && hasEffectResult(result)) {
                text = formatEffectResult({...result, ok: false, error: result.error || failure});
            } else if (failed) {
                text = failure;
            } else if (completed) {
                text = formatEffectResult(result);
            } else if (transportError) {
                text += `\n${transportError}`;
            }
            replaceApprovalWithResult(card, text, {error: failed || !completed || result?.ok === false});
            setBusy(false);
            refreshSessions({quiet: true});
        }
    }

    async function cancelEffect(button) {
        if (busy) return;
        const card = button.closest('.agent-confirmation-card');
        const planId = button.dataset.effectCancel || '';
        if (!card || !planId) return;
        card.querySelectorAll('button').forEach((item) => { item.disabled = true; });
        try {
            const payload = await fetchJSON('/api/agent/actions/confirm/discard', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({plan_id: planId, session_id: sessionId, request_id: createId('cancel')}),
            });
            replaceApprovalWithResult(
                card,
                payload.discarded ? '本次计划已取消，没有执行任何写操作。' : '该确认已过期或已处理。',
                {cancelled: true},
            );
        } catch (error) {
            replaceApprovalWithResult(card, error?.message || '暂时无法取消该计划。', {error: true});
        } finally {
            refreshSessions({quiet: true});
        }
    }

    function sessionTime(value) {
        const numeric = Number(value);
        const date = Number.isFinite(numeric) ? new Date(numeric * 1000) : new Date(value);
        if (Number.isNaN(date.getTime())) return '';
        return new Intl.DateTimeFormat('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'}).format(date);
    }

    function renderSessionList(items) {
        sessionItems = (Array.isArray(items) ? items : []).filter((item) => SESSION_RE.test(String(item?.session_id || ''))).slice(0, 100)
            .map((item) => ({...item, pinned: item.pinned === true}));
        if (!sessionList) return;
        const newest = [...sessionItems].sort((a, b) => (Number(b.updated_at) || 0) - (Number(a.updated_at) || 0));
        latestSessionId = newest[0]?.session_id || '';
        if (resumeButton) resumeButton.disabled = busy || !latestSessionId;
        const query = String(sessionSearch?.value || '').trim().normalize('NFKC').toLocaleLowerCase();
        const sorted = [...sessionItems].sort((a, b) => Number(b.pinned) - Number(a.pinned) ||
            (Number(b.updated_at) || 0) - (Number(a.updated_at) || 0));
        const existing = new Map([...sessionList.querySelectorAll('.agent-session-item')].map(row => [row.dataset.sessionId, row]));
        const ids = new Set(sorted.map(item => item.session_id));
        const scrollTop = sessionList.scrollTop;
        const focused = sessionList.contains(document.activeElement) ? document.activeElement : null;
        let shown = 0;
        for (const [id, row] of existing) if (!ids.has(id)) row.remove();
        for (const item of sorted) {
            let row = existing.get(item.session_id);
            if (!row) {
                row = element('div', 'agent-session-item');
                row.dataset.sessionId = item.session_id;
                const open = element('button', 'agent-session-open');
                open.type = 'button';
                open.dataset.sessionOpen = item.session_id;
                open.append(element('strong', ''), element('small', ''));
                const controls = element('div', 'agent-session-controls');
                for (const [name, mark, label] of [['pin', 'pin', '置顶'], ['rename', 'pencil', '重命名'], ['delete', 'trash-2', '删除']]) {
                    const button = element('button', `agent-session-${name}`);
                    button.type = 'button';
                    button.dataset[`session${name[0].toUpperCase()}${name.slice(1)}`] = item.session_id;
                    button.title = label;
                    button.append(icon(mark));
                    controls.append(button);
                }
                row.append(open, controls);
                renderIcons(row);
            }
            const title = String(item.title || '新对话');
            const open = row.querySelector('.agent-session-open');
            open.querySelector('strong').textContent = title;
            open.querySelector('small').textContent = `${item.pinned ? '置顶 · ' : ''}${item.message_count || 0} 条消息${sessionTime(item.updated_at) ? ` · ${sessionTime(item.updated_at)}` : ''}`;
            open.title = title;
            row.classList.toggle('is-active', item.session_id === sessionId);
            const pin = row.querySelector('[data-session-pin]');
            pin.setAttribute('aria-pressed', String(item.pinned));
            pin.setAttribute('aria-label', `${item.pinned ? '取消置顶' : '置顶'}会话 ${title}`);
            pin.title = item.pinned ? '取消置顶' : '置顶';
            row.querySelector('[data-session-rename]').setAttribute('aria-label', `重命名会话 ${title}`);
            row.querySelector('[data-session-delete]').setAttribute('aria-label', `删除会话 ${title}`);
            row.hidden = Boolean(query && !title.normalize('NFKC').toLocaleLowerCase().includes(query));
            if (!row.hidden) shown += 1;
            sessionList.append(row);
        }
        let empty = sessionList.querySelector('.agent-session-empty');
        if (!shown) {
            if (!empty) {
                empty = element('div', 'agent-session-empty');
                empty.append(icon('message-circle-dashed'), element('span', ''));
                sessionList.append(empty);
                renderIcons(empty);
            }
            empty.querySelector('span').textContent = query ? '没有匹配的会话，试试其他标题' : '尚无已保存的对话';
        } else empty?.remove();
        if (sessionCount) sessionCount.textContent = `${shown} 条`;
        sessionList.scrollTop = scrollTop;
        if (focused?.isConnected && !focused.closest('[hidden]') && document.activeElement !== focused) focused.focus({preventScroll: true});
    }

    function editSessionTitle(id) {
        if (sessionEdits.has(id)) return;
        const item = sessionItems.find(item => item.session_id === id);
        const row = [...sessionList.querySelectorAll('.agent-session-item')].find(row => row.dataset.sessionId === id);
        if (!item || !row || row.querySelector('form')) return;
        const form = element('form', 'agent-session-editor');
        const input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 80;
        input.value = String(item.title || '');
        input.setAttribute('aria-label', '会话名称');
        const save = element('button', '', '保存');
        save.type = 'submit';
        const cancel = element('button', '', '取消');
        cancel.type = 'button';
        const close = () => {
            const restoreFocus = form.contains(document.activeElement) || document.activeElement === document.body;
            form.remove();
            row.classList.remove('is-editing');
            if (restoreFocus && historyRail?.open) row.querySelector('[data-session-rename]')?.focus({preventScroll: true});
        };
        cancel.addEventListener('click', close);
        input.addEventListener('keydown', event => {
            if (event.key === 'Escape') { event.stopPropagation(); event.preventDefault(); close(); }
        });
        form.addEventListener('submit', async event => {
            event.preventDefault();
            const title = input.value.trim();
            if (!title) { announce(sessionStatus, '会话名称不能为空'); input.focus(); return; }
            if (await patchSession(id, {title})) close();
        });
        form.append(input, save, cancel);
        row.classList.add('is-editing');
        row.append(form);
        input.focus();
        input.select();
    }

    async function patchSession(id, values) {
        if (sessionEdits.has(id)) return false;
        sessionEdits.add(id);
        const row = [...sessionList.querySelectorAll('.agent-session-item')].find(row => row.dataset.sessionId === id);
        const controls = [...(row?.querySelectorAll('button,input') || [])];
        const focused = controls.includes(document.activeElement) ? document.activeElement : null;
        const selection = focused && typeof focused.selectionStart === 'number'
            ? [focused.selectionStart, focused.selectionEnd] : null;
        controls.forEach(control => { control.disabled = true; });
        try {
            const payload = await fetchJSON(`/api/agent/sessions/${encodeURIComponent(id)}`, {
                method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(values),
            });
            const updated = payload.session;
            if (!updated || updated.session_id !== id) throw new Error('会话更新结果无效，请刷新列表核验');
            sessionItems = sessionItems.map(item => item.session_id === id ? {...item, ...updated} : item);
            renderSessionList(sessionItems);
            announce(sessionStatus, '会话已更新');
            return true;
        } catch (error) {
            announce(sessionStatus, error?.message || '会话更新失败，请稍后重试');
            return false;
        } finally {
            sessionEdits.delete(id);
            controls.forEach(control => { control.disabled = false; });
            // disabled 会使键盘焦点落到 body；失败后恢复，但不抢走用户主动移到别处的焦点。
            if (focused?.isConnected && historyRail?.open &&
                (document.activeElement === document.body || document.activeElement === focused)) {
                focused.focus({preventScroll: true});
                if (selection && typeof focused.setSelectionRange === 'function') focused.setSelectionRange(...selection);
            }
        }
    }

    async function refreshNextActions() {
        if (!nextActions) return;
        nextActions.setAttribute('aria-busy', 'true');
        try {
            const payload = await fetchJSON('/api/agent/next-actions');
            const actions = (Array.isArray(payload.actions) ? payload.actions : []).filter(item =>
                typeof item?.title === 'string' && typeof item.prompt === 'string' && item.prompt.trim()).slice(0, 3);
            const nodes = [];
            for (const action of actions) {
                const button = element('button', 'agent-start-action');
                button.type = 'button';
                button.dataset.agentDraft = clipText(action.prompt, 1000);
                button.title = String(action.description || action.title).slice(0, 300);
                button.append(element('span', '', action.title.slice(0, 80)), icon('arrow-up-right'));
                nodes.push(button);
            }
            nextActions.replaceChildren(...nodes);
            if (nextActionsStatus) nextActionsStatus.textContent = actions.length
                ? '可以从这里开始 · 仅查看，不自动处理'
                : (payload.snapshot_status === 'unavailable' ? '待办暂时不可用，仍可直接提问' : '暂时没有待处理事项，也可以直接提问');
            renderIcons(nextActions);
        } catch (_) {
            if (nextActionsStatus) nextActionsStatus.textContent = '待办暂时不可用，仍可直接提问';
        } finally { nextActions.setAttribute('aria-busy', 'false'); }
    }

    async function refreshSessions({quiet = false} = {}) {
        historyController?.abort();
        const controller = new AbortController();
        historyController = controller;
        if (!quiet) sessionList?.setAttribute('aria-busy', 'true');
        try {
            const payload = await fetchJSON('/api/agent/sessions', {signal: controller.signal});
            if (historyController !== controller) return;
            configureDraftScope(payload.draft_scope);
            renderSessionList(payload.sessions || []);
            announce(sessionStatus, '会话列表已更新');
        } catch (error) {
            if (error?.name !== 'AbortError' && !quiet) announce(sessionStatus, '会话列表加载失败');
        } finally {
            if (historyController === controller) {
                historyController = null;
                sessionList?.setAttribute('aria-busy', 'false');
            }
        }
    }

    function renderRecoveredApproval(approval) {
        if (!approval?.plan_id) return;
        const view = appendMessage('assistant', {recovered: true});
        view.body.append(buildApproval(approval));
    }

    async function loadSession(targetId, {closeHistory = true} = {}) {
        if (busy || !SESSION_RE.test(targetId)) return;
        const generation = ++sessionLoadGeneration;
        try {
            const payload = await fetchJSON(`/api/agent/sessions/${encodeURIComponent(targetId)}`);
            if (generation !== sessionLoadGeneration) return;
            saveDraft();
            rememberSession(targetId);
            restoreDraft();
            expireCandidateCards();
            transcript?.replaceChildren();
            for (const message of payload.messages || []) {
                if (message.role === 'user') appendUser(String(message.content || ''), {recovered: true});
                else if (message.role === 'assistant') {
                    const turn = createAssistantTurn({recovered: true});
                    addRecoveredToolTrace(turn, message.tools, message.tool_labels);
                    finalizeAnswer(turn, String(message.content || ''));
                }
            }
            if (payload.candidate_view) {
                const view = appendMessage('assistant', {recovered: true});
                renderCandidateView({card: view.body}, payload.candidate_view);
            }
            renderRecoveredApproval(payload.pending_approval);
            followOutput = true;
            scrollToBottom(true);
            setConsoleEmpty(!transcript?.childElementCount);
            if (closeHistory) closeHistoryRail();
            refreshSessions({quiet: true});
        } catch (error) {
            announce(sessionStatus, error?.message || '会话加载失败');
        }
    }

    async function deleteSession(targetId) {
        if (busy || !SESSION_RE.test(targetId)) return;
        try {
            await fetchJSON(`/api/agent/sessions/${encodeURIComponent(targetId)}`, {method: 'DELETE'});
            removeDraft(targetId);
            if (targetId === sessionId) {
                promptInput.value = '';
                startNewSession();
            }
            await refreshSessions();
        } catch (error) {
            announce(sessionStatus, error?.message || '会话删除失败');
        }
    }

    function startNewSession() {
        if (busy) return;
        ++sessionLoadGeneration;
        saveDraft();
        expireCandidateCards();
        rememberSession(createId('session'));
        restoreDraft();
        followOutput = true;
        if (newRepliesButton) newRepliesButton.hidden = true;
        transcript?.replaceChildren();
        setConsoleEmpty(true);
        promptInput?.focus();
        closeHistoryRail();
        refreshSessions({quiet: true});
    }

    function openHistoryRail() {
        if (!historyRail) return;
        if (typeof historyRail.showModal === 'function') {
            if (!historyRail.open) historyRail.showModal();
        } else {
            historyRail.setAttribute('open', '');
        }
        historyButton?.setAttribute('aria-expanded', 'true');
        document.getElementById('agent-session-heading')?.focus({preventScroll: true});
        refreshSessions();
    }

    function closeHistoryRail() {
        if (!historyRail) return;
        if (typeof historyRail.close === 'function' && historyRail.open) historyRail.close();
        else historyRail.removeAttribute('open');
        historyButton?.setAttribute('aria-expanded', 'false');
    }

    function expireCandidateCards() {
        if (candidateExpiryTimer !== null) clearTimeout(candidateExpiryTimer);
        candidateExpiryTimer = null;
        transcript?.querySelectorAll('[data-candidate-select]').forEach((button) => {
            button.disabled = true;
            button.dataset.expired = 'true';
            button.title = '候选已更新，请基于新的搜索结果选择';
        });
        transcript?.querySelectorAll('.agent-candidates-note').forEach(note => {
            note.textContent = '此批候选仅供回看，请基于新的搜索结果选择。';
        });
    }

    function syncCandidateButtons() {
        transcript?.querySelectorAll('[data-candidate-select]').forEach(button => {
            const expired = button.dataset.expired === 'true' || Number(button.dataset.expiresAt) * 1000 <= Date.now();
            button.disabled = busy || expired;
            if (expired) {
                button.title = '候选已过期或更新，请重新搜索';
                const note = button.closest('.agent-candidates')?.querySelector('.agent-candidates-note');
                if (note) note.textContent = '此批候选仅供回看，请基于新的搜索结果选择。';
            }
        });
    }

    function renderCandidateView(turn, view) {
        if (!view || !Array.isArray(view.items) || typeof view.ref !== 'string' || !Number.isFinite(view.expires_at)) return;
        if (turn.candidateGroup?.dataset.candidateView === view.ref) return;
        const items = view.items.filter(item => typeof item?.title === 'string' && Number.isInteger(item.position) &&
            item.position > 0 && item.position <= 12 && typeof item.selection?.ref === 'string' &&
            /^ref_[A-Za-z0-9_-]{16,160}$/.test(item.selection.ref) && item.selection.position === item.position).slice(0, 12);
        if (!items.length) return;
        expireCandidateCards();
        const group = element('section', 'agent-candidates');
        group.dataset.candidateView = view.ref;
        group.setAttribute('aria-label', '比较资源候选');
        const heading = element('div', 'agent-candidates-heading');
        heading.append(element('strong', '', '资源候选'), element('span', '', `${items.length} 项可供核对`));
        const note = element('p', 'agent-candidates-note', '选择后先生成预览，确认后才提交下载。');
        const grid = element('div', 'agent-candidate-grid');
        const extra = document.createElement('details');
        extra.className = 'agent-candidates-more';
        const summary = document.createElement('summary');
        summary.textContent = `查看另外 ${Math.max(0, items.length - 4)} 项候选`;
        const moreGrid = element('div', 'agent-candidate-grid');
        extra.append(summary, moreGrid);
        const tagNames = {resolution: '画质', media: '版本', video_codec: '编码', effect: '画面', audio: '音轨'};
        for (const [index, item] of items.entries()) {
            const card = element('article', 'agent-candidate-card');
            const meta = element('p', 'agent-candidate-meta', `#${item.position}${item.site_name ? ` · ${String(item.site_name).slice(0, 80)}` : ''}${item.size_text ? ` · ${String(item.size_text).slice(0, 32)}` : ''}`);
            const title = element('h4', '', item.title.slice(0, 300));
            title.title = item.title.slice(0, 300);
            const tags = element('div', 'agent-candidate-tags');
            for (const [key, label] of Object.entries(tagNames)) {
                const value = item.tags?.[key];
                if (typeof value === 'string' && value.trim()) tags.append(element('span', '', `${label} · ${value.slice(0, 64)}`));
            }
            const reasons = element('ul', 'agent-candidate-reasons');
            for (const reason of (Array.isArray(item.reasons) ? item.reasons : []).filter(value => typeof value === 'string').slice(0, 3)) {
                reasons.append(element('li', '', reason.slice(0, 120)));
            }
            const warnings = element('ul', 'agent-candidate-warnings');
            for (const warning of (Array.isArray(item.warnings) ? item.warnings : []).filter(value => typeof value === 'string').slice(0, 4)) {
                warnings.append(element('li', '', warning.slice(0, 120)));
            }
            const button = element('button', 'agent-candidate-select', '选择并预览');
            button.type = 'button';
            button.dataset.candidateSelect = item.selection.ref;
            button.dataset.candidatePosition = String(item.selection.position);
            button.dataset.candidateTitle = clipText(item.title, 160);
            button.dataset.expiresAt = String(view.expires_at);
            button.setAttribute('aria-label', `选择候选 ${item.position} 并预览`);
            card.append(meta, title);
            if (tags.childElementCount) card.append(tags);
            if (reasons.childElementCount) card.append(reasons);
            if (warnings.childElementCount) card.append(warnings);
            card.append(button);
            (index < 4 ? grid : moreGrid).append(card);
        }
        group.append(heading, note, grid);
        if (items.length > 4) group.append(extra);
        turn.candidateGroup = group;
        turn.card.append(group);
        syncCandidateButtons();
        candidateExpiryTimer = setTimeout(syncCandidateButtons, Math.max(0, Math.min(2147483647, view.expires_at * 1000 - Date.now() + 25)));
        scrollToBottom();
    }

    function selectCandidate(button) {
        if (busy || button.disabled) return;
        if (button.dataset.expired === 'true' || Number(button.dataset.expiresAt) * 1000 <= Date.now()) {
            syncCandidateButtons();
            announce(responseStatus, '候选已过期，请重新搜索后选择');
            window.showToast?.('候选已过期，请重新搜索后选择', 'warning');
            return;
        }
        const selection = {ref: button.dataset.candidateSelect, position: Number(button.dataset.candidatePosition)};
        sendQuery(`选择候选 #${selection.position}「${button.dataset.candidateTitle || ''}」并生成下载预览。`, {selection, preserveDraft: true});
    }

    function syncViewportHeight() {
        const height = window.visualViewport?.height || window.innerHeight;
        document.documentElement.style.setProperty('--agent-viewport-height', `${Math.round(height)}px`);
        consoleNode?.style.setProperty('--agent-composer-height', `${Math.round(composer?.getBoundingClientRect().height || 100)}px`);
    }

    composer?.addEventListener('submit', (event) => {
        event.preventDefault();
        sendQuery(promptInput?.value || '');
    });
    promptInput?.addEventListener('input', () => { resizePrompt(); saveDraft(); });
    transcript?.addEventListener('scroll', () => {
        followOutput = transcriptNearBottom();
        if (followOutput && newRepliesButton) newRepliesButton.hidden = true;
    }, {passive: true});
    newRepliesButton?.addEventListener('click', () => scrollToBottom(true));
    window.addEventListener('pagehide', saveDraft);
    promptInput?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            composer?.requestSubmit();
        }
    });
    stopButton?.addEventListener('click', stopActiveRequest);
    newSessionButton?.addEventListener('click', startNewSession);
    resumeButton?.addEventListener('click', () => latestSessionId && loadSession(latestSessionId));
    historyButton?.addEventListener('click', openHistoryRail);
    historyRail?.addEventListener('cancel', (event) => {
        event.preventDefault();
        closeHistoryRail();
    });
    historyRail?.addEventListener('click', (event) => {
        if (event.target === historyRail || event.target.closest('[data-agent-history-close]')) closeHistoryRail();
    });
    sessionSearch?.addEventListener('input', () => renderSessionList(sessionItems));
    sessionList?.addEventListener('click', (event) => {
        const open = event.target.closest('[data-session-open]');
        const remove = event.target.closest('[data-session-delete]');
        const rename = event.target.closest('[data-session-rename]');
        const pin = event.target.closest('[data-session-pin]');
        if (rename) editSessionTitle(rename.dataset.sessionRename);
        if (pin) {
            const item = sessionItems.find(item => item.session_id === pin.dataset.sessionPin);
            if (item) patchSession(item.session_id, {pinned: !item.pinned});
        }
        if (open) loadSession(open.dataset.sessionOpen || '');
        if (remove) deleteSession(remove.dataset.sessionDelete || '');
    });
    transcript?.addEventListener('click', (event) => {
        const candidate = event.target.closest('[data-candidate-select]');
        if (candidate) selectCandidate(candidate);
        const confirm = event.target.closest('[data-effect-confirm]');
        const cancel = event.target.closest('[data-effect-cancel]');
        if (confirm) confirmEffect(confirm);
        if (cancel) cancelEffect(cancel);
    });
    page.addEventListener('click', (event) => {
        const draft = event.target.closest('[data-agent-draft]');
        if (draft) fillDraft(draft.dataset.agentDraft);
    });
    window.visualViewport?.addEventListener('resize', syncViewportHeight, {passive: true});
    window.addEventListener('resize', syncViewportHeight, {passive: true});

    syncViewportHeight();
    resizePrompt();
    setConsoleEmpty(true);
    refreshNextActions();
    renderIcons(page);
    refreshSessions({quiet: true}).then(() => {
        if (storedSessionId()) loadSession(sessionId, {closeHistory: false});
    });
})();
