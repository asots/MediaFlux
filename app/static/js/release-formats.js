(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const modal = $('releaseFormatsModal');
    const openButton = $('openReleaseFormatsBtn');
    if (!modal || !openButton) return;

    const workbench = window.createResponsiveRulesWorkbench(modal);

    const refs = {
        form: $('releaseFormatsForm'),
        name: $('releaseFormatName'),
        scope: $('releaseFormatScope'),
        parentPath: $('releaseFormatParentPath'),
        parentField: $('releaseFormatParentPathField'),
        directoryPick: $('releaseFormatPickDirectoryBtn'),
        directoryPickStart: $('releaseFormatPickStartBtn'),
        directoryStart: $('releaseFormatDirectoryStart'),
        directoryOrigin: $('releaseFormatDirectoryOrigin'),
        directorySource: $('releaseFormatDirectorySource'),
        effectivePath: $('releaseFormatEffectivePath'),
        contextOrigin: $('releaseFormatContextOrigin'),
        contextExplanation: $('releaseFormatContextExplanation'),
        scopeNote: $('releaseFormatScopeNote'),
        releaseNote: $('releaseFormatReleaseNote'),
        template: $('releaseFormatTemplate'),
        filenames: $('releaseFormatFilenames'),
        examples: $('releaseFormatExamples'),
        exampleTemplate: $('releaseFormatExampleTemplate'),
        addExample: $('addReleaseFormatExampleBtn'),
        loadExample: $('loadReleaseFormatExampleBtn'),
        preview: $('previewReleaseFormatBtn'),
        save: $('saveReleaseFormatBtn'),
        formState: $('releaseFormatsFormState'),
        list: $('releaseFormatsList'),
        listState: $('releaseFormatsListState'),
        summary: $('releaseFormatsSummary'),
        previewState: $('releaseFormatsPreviewState'),
        ticket: $('releaseFormatsTicketState'),
        frame: $('releaseFormatPreviewFrame'),
        empty: $('releaseFormatPreviewEmpty'),
        tableWrap: $('releaseFormatPreviewTableWrap'),
        table: $('releaseFormatPreviewTable'),
        warnings: $('releaseFormatWarnings'),
        filenameCount: $('releaseFormatFilenameCount'),
        editorTitle: $('releaseFormatsEditorTitle'),
        ruleSummary: $('releaseFormatRuleSummary'),
    };

    const MIN_EXAMPLES = 2;
    const MAX_EXAMPLES = 8;
    const MAX_FILENAMES = 100;
    const FIELDS = ['title', 'episode', 'season', 'version', 'resolution', 'checksum'];
    const TOKENS = Object.fromEntries(FIELDS.map((field) => [field, `{${field}}`]));
    const STATUS_LABELS = {
        matched: '命中', unchanged: '无变化', unmatched: '未匹配',
        blocked: '已阻止', conflict: '冲突',
    };
    const SCOPE_LABELS = {
        directory: '适用文件夹 · 仅此文件夹（不含子文件夹）',
        release: '发布组 · 跨作品',
    };
    const TEACHING_EXAMPLE = {
        name: 'Example-Team 教学示例',
        template: '[Example-Team][{title}][track{episode}r{version}][{resolution}].mkv',
        scope: 'directory',
        parent_path: '/Anime/Teaching',
        examples: [
            {filename: '[Example-Team][星海航行][track013r2][1080p].mkv', title: '星海航行', episode: 13},
            {filename: '[Example-Team][星海航行][track014r2][1080p].mkv', title: '星海航行', episode: 14},
        ],
        filenames: [
            '[Example-Team][星海航行][track013r2][1080p].mkv',
            '[Example-Team][星海航行][track014r2][1080p].mkv',
        ],
    };

    const state = {
        rules: [],
        rulesLoaded: false,
        rulesRequestSerial: 0,
        inputVersion: 0,
        inputFingerprint: '',
        previewSerial: 0,
        previewBusy: false,
        saveBusy: false,
        preview: null,
        templateSelection: {start: 0, end: 0},
        directorySources: {guangya: [], local: []},
        directorySourcesBusy: false,
        directorySourceRequestSerial: 0,
        directoryPickerSerial: 0,
        directorySelectionApplying: false,
        directoryContext: null,
        directoryStart: null,
    };

    const lifecycle = window.createAppModal(modal, {onRequestClose: ({close}) => {
        invalidateDirectoryPicker();
        state.directorySourceRequestSerial += 1;
        state.directorySourcesBusy = false;
        close();
    }});

    const iconize = (root = modal) => window.renderLucideIcons(root);
    const text = (node, value) => { if (node) node.textContent = value; };
    const node = (tag, className, value) => {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (value !== undefined) element.textContent = value;
        return element;
    };
    function ruleRevision(rule) {
        if (!Number.isInteger(rule?.revision) || rule.revision < 1) throw new Error('规则响应缺少有效 revision。');
        return rule.revision;
    }
    function requireRuleItem(item) {
        if (!item || !Number.isInteger(item.id) || typeof item.name !== 'string'
            || typeof item.template !== 'string' || !['directory', 'release'].includes(item.scope)
            || typeof item.parent_path !== 'string' || !Array.isArray(item.examples)
            || item.examples.some((example) => !example || typeof example.filename !== 'string')
            || typeof item.disabled !== 'boolean') {
            throw new Error('规则响应格式无效。');
        }
        ruleRevision(item);
        return item;
    }
    const setMessage = (value, kind = '') => {
        text(refs.formState, value);
        refs.formState?.classList.toggle('is-error', kind === 'error');
        refs.formState?.classList.toggle('is-success', kind === 'success');
    };
    const setTicket = (value, ready = false) => {
        text(refs.ticket, value);
        refs.ticket?.classList.toggle('is-ready', ready);
    };
    const csrfToken = () => document.querySelector('meta[name="csrf-token"]')?.content || '';

    async function requestJson(path, {method = 'GET', body, signal} = {}) {
        const headers = new Headers({Accept: 'application/json'});
        if (body !== undefined) headers.set('Content-Type', 'application/json');
        if (!['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method)) headers.set('X-CSRF-Token', csrfToken());
        const response = await fetch(path, {
            credentials: 'same-origin', method, headers,
            ...(body === undefined ? {} : {body}),
            ...(signal ? {signal} : {}),
        });
        let data = {};
        try { data = await response.json(); } catch (_) {}
        if (!response.ok) {
            const error = new Error(String(data.error || `请求失败 (${response.status})`));
            error.status = response.status;
            throw error;
        }
        return data;
    }

    function normalizeDirectorySource(item, origin) {
        if (!item || !item.id || !item.name || String(item.id) === '0') return null;
        const rootId = origin === 'local' ? String(item.local_root || '').replace(/\\/g, '/').replace(/\/+$/, '') || '/' : String(item.id);
        if (origin === 'local' && !item.local_root) return null;
        return {
            id: String(item.id), label: String(item.name), rootId,
            name: origin === 'local' ? rootId.split('/').filter(Boolean).at(-1) || '' : String(item.name),
            enabled: item.enabled !== false,
        };
    }

    function currentDirectoryOrigin() { return refs.directoryOrigin.value; }
    function selectedDirectorySource() {
        return state.directorySources[currentDirectoryOrigin()].find(item => item.id === refs.directorySource.value) || null;
    }
    function selectedStart(source = selectedDirectorySource()) {
        if (!source) return null;
        const selected = state.directoryStart;
        return selected?.origin === currentDirectoryOrigin() && selected.sourceId === source.id && selected.sourceRoot === source.rootId
            ? selected : {id: source.rootId, name: source.name};
    }
    function invalidateDirectoryPicker() { state.directoryPickerSerial += 1; }

    function syncDirectoryContext() {
        const parent = refs.parentPath.value.trim();
        const start = selectedStart();
        text(refs.directoryStart, start?.name || '请选择具体文件夹作为整理起点');
        refs.directoryStart.title = start?.name || '';
        text(refs.effectivePath, parent || '选择文件夹后自动填入');
        const context = state.directoryContext;
        text(refs.contextOrigin, context?.parentPath === parent ? context.startName : '已有或手动填写的路径');
        text(refs.contextExplanation, context?.parentPath === parent
            ? `从「${context.startName}」开始整理时生效；仅匹配「${parent}」中的文件，不包含子文件夹。`
            : '已填路径保持原样。请选择本次整理起点和文件所在目录来更新；浏览来源本身不会改写规则。');
        syncDirectoryPickerControls();
    }
    function syncDirectoryPickerControls() {
        const disabled = refs.scope.value === 'release' || state.directorySourcesBusy || !selectedDirectorySource();
        refs.directoryPick.disabled = disabled;
        refs.directoryPickStart.disabled = disabled;
    }
    async function refreshDirectorySources() {
        const origin = currentDirectoryOrigin();
        const previous = refs.directorySource.value;
        const serial = ++state.directorySourceRequestSerial;
        state.directorySourcesBusy = true;
        refs.directorySource.disabled = true;
        syncDirectoryPickerControls();
        let error = '';
        try {
            let values;
            if (origin === 'guangya') {
                const ready = await window.organizeConfigReady;
                if (ready?.success === false) throw new Error('光鸭整理来源尚未就绪，请刷新页面重试');
                values = window.getOrganizeSourceDirectories?.() || [];
            } else {
                const data = await requestJson('/api/local-media/sources');
                if (!Array.isArray(data.sources)) throw new Error('本地来源响应无效，请重试');
                values = data.sources;
            }
            if (serial !== state.directorySourceRequestSerial) return;
            state.directorySources[origin] = values.map(item => normalizeDirectorySource(item, origin)).filter(item => item?.enabled);
        } catch (failure) {
            if (serial !== state.directorySourceRequestSerial) return;
            state.directorySources[origin] = [];
            error = failure.message || '来源读取失败，请重试';
        } finally {
            if (serial === state.directorySourceRequestSerial) {
                state.directorySourcesBusy = false;
                const sources = state.directorySources[origin];
                refs.directorySource.replaceChildren();
                for (const source of sources) {
                    const option = node('option', '', source.label);
                    option.value = source.id;
                    refs.directorySource.append(option);
                }
                if (!sources.length) {
                    const option = node('option', '', error || '暂无来源，请先在对应整理页面配置');
                    option.value = '';
                    refs.directorySource.append(option);
                } else {
                    refs.directorySource.value = sources.some(item => item.id === previous) ? previous : sources[0].id;
                }
                refs.directorySource.disabled = !sources.length;
                syncDirectoryContext();
            }
        }
    }

    async function pickDirectory(mode) {
        if (refs.scope.value === 'release' || state.directorySourcesBusy) return;
        const serial = ++state.directoryPickerSerial;
        const origin = currentDirectoryOrigin();
        const sourceId = refs.directorySource.value;
        await refreshDirectorySources();
        if (serial !== state.directoryPickerSerial || currentDirectoryOrigin() !== origin || refs.directorySource.value !== sourceId) return;
        const source = selectedDirectorySource();
        if (!source) return;
        if (typeof window.openGuangYaDirectoryPicker !== 'function') {
            setMessage('目录选择器尚未加载，请刷新页面；已有路径仍可手动填写。', 'error');
            return;
        }
        const root = mode === 'start' ? {id: source.rootId, name: source.name} : selectedStart(source);
        const options = {
            modalId: 'releaseFormatDirectoryModal',
            title: mode === 'start' ? '选择本次开始整理的文件夹' : '选择文件所在的文件夹',
            rootId: root.id, rootName: root.name || '/', allowRoot: Boolean(root.name), preserveWhileLoading: true,
            onSelect: directory => {
                if (serial !== state.directoryPickerSerial || refs.scope.value !== 'directory'
                    || currentDirectoryOrigin() !== origin || selectedDirectorySource()?.id !== source.id) return false;
                if (!Array.isArray(directory?.path) || !directory.name) return false;
                const start = mode === 'start' ? directory : root;
                const parent = [start.name, ...(mode === 'start' ? [] : directory.path.map(part => part.name))].filter(Boolean).join('/');
                if (!parent) return false;
                if (mode === 'start') state.directoryStart = {origin, sourceId: source.id, sourceRoot: source.rootId, id: start.id, name: start.name};
                state.directoryContext = {startName: start.name, parentPath: parent};
                state.directorySelectionApplying = true;
                try {
                    refs.parentPath.value = parent;
                    refs.parentPath.dispatchEvent(new Event('input', {bubbles: true}));
                } finally { state.directorySelectionApplying = false; }
                syncDirectoryContext();
                setMessage('适用路径已自动填入，请刷新预览后确认保存。');
                return true;
            },
        };
        if (origin === 'local') options.fetchDirectory = async (path, {signal} = {}) => {
            const query = new URLSearchParams({source_id: source.id, path});
            const data = await requestJson(`/api/local-media/directories?${query}`, {signal});
            if (!Array.isArray(data.directories)) throw new Error('本地目录响应无效');
            return data.directories.map(item => ({id: item.path, name: item.name, is_dir: true}));
        };
        window.openGuangYaDirectoryPicker(options);
    }

    function setInput(row, field, value) {
        const input = row.querySelector(`[data-example-field="${field}"]`);
        if (input) input.value = value === undefined || value === null ? '' : String(value);
    }

    function exampleRows() {
        return [...refs.examples.querySelectorAll('[data-example-row]')];
    }

    function buildExampleRow(example = {}) {
        const row = refs.exampleTemplate.content.firstElementChild.cloneNode(true);
        ['filename', 'title', 'episode', 'season'].forEach((field) => setInput(row, field, example[field]));
        return row;
    }

    function clearExampleResults() {
        exampleRows().forEach((row) => {
            const result = row.querySelector('[data-example-result]');
            if (!result) return;
            result.textContent = '待预览';
            result.classList.remove('is-passed', 'is-failed');
        });
    }

    function syncExampleControls() {
        const rows = exampleRows();
        rows.forEach((row, index) => {
            text(row.querySelector('[data-example-label]'), `样本 ${String(index + 1).padStart(2, '0')}`);
            const remove = row.querySelector('[data-remove-example]');
            if (!remove) return;
            remove.disabled = rows.length <= MIN_EXAMPLES;
            remove.title = remove.disabled ? '至少保留两条样本' : '移除样本';
            remove.setAttribute('aria-label', `移除样本 ${index + 1}`);
        });
        refs.addExample.disabled = rows.length >= MAX_EXAMPLES;
        refs.addExample.title = refs.addExample.disabled ? '最多 8 条样本' : '添加样本';
    }

    function renderExampleRows(values) {
        const examples = Array.isArray(values) ? values.slice(0, MAX_EXAMPLES) : [];
        while (examples.length < MIN_EXAMPLES) examples.push({});
        refs.examples.replaceChildren(...examples.map(buildExampleRow));
        syncExampleControls();
        clearExampleResults();
        iconize(refs.examples);
    }

    function readExamples() {
        return exampleRows().map((row) => {
            const value = {
                filename: row.querySelector('[data-example-field="filename"]')?.value.trim() || '',
                title: row.querySelector('[data-example-field="title"]')?.value.trim() || '',
                episode: row.querySelector('[data-example-field="episode"]')?.value.trim() || '',
            };
            const season = row.querySelector('[data-example-field="season"]')?.value.trim() || '';
            if (season) value.season = season;
            return value;
        });
    }

    const number = (value) => {
        if (value === '' || value === undefined || value === null) return null;
        const result = Number(value);
        return Number.isInteger(result) ? result : NaN;
    };

    function normalizeExamples() {
        return readExamples().map((example) => ({
            filename: example.filename,
            title: example.title,
            episode: number(example.episode),
            ...(example.season === undefined ? {} : {season: number(example.season)}),
        }));
    }

    function readFilenames() {
        return String(refs.filenames.value || '').split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
    }

    function readPayload() {
        const scope = refs.scope.value === 'release' ? 'release' : 'directory';
        const draft = {
            name: refs.name.value.trim(),
            template: refs.template.value,
            scope,
            parent_path: scope === 'directory' ? refs.parentPath.value.trim() : '',
        };
        return {draft, examples: normalizeExamples(), filenames: readFilenames()};
    }

    const fingerprint = (payload) => JSON.stringify(payload);

    function collectInput() {
        const payload = readPayload();
        const {draft, examples, filenames} = payload;
        const errors = [];
        const slots = [...draft.template.matchAll(/\{([^{}]+)\}/g)].map((match) => match[1]);
        const remainder = draft.template.replace(/\{[^{}]+\}/g, '');

        if (!draft.name) errors.push('请填写规则名称。');
        if (!draft.template.trim()) errors.push('请填写字段模板。');
        if (/[{}]/.test(remainder)) errors.push('模板中的字段格式无效，请使用下方字段按钮。');
        const unknown = slots.find((field) => !FIELDS.includes(field));
        const duplicate = slots.find((field, index) => slots.indexOf(field) !== index);
        if (unknown) errors.push(`模板字段“{${unknown}}”不受支持。`);
        if (duplicate) errors.push(`模板字段“{${duplicate}}”不能重复。`);
        if (!slots.includes('title')) errors.push('模板必须包含 {title}。');
        if (!slots.includes('episode')) errors.push('模板必须包含 {episode}。');
        if (/\}\s*\{/.test(draft.template)) errors.push('字段不能相邻无分隔，请在字段之间加入字面量分隔符。');
        if (draft.scope === 'directory' && !draft.parent_path) errors.push('请选择或填写适用文件夹。');
        if (examples.length < MIN_EXAMPLES || examples.length > MAX_EXAMPLES) errors.push('样本数量必须在 2 到 8 条之间。');

        examples.forEach((example, index) => {
            const label = `样本 ${String(index + 1).padStart(2, '0')}`;
            if (!example.filename) errors.push(`${label} 需要填写 filename。`);
            if (!example.title) errors.push(`${label} 需要填写 title。`);
            if (!Number.isInteger(example.episode) || example.episode < 1 || example.episode > 9999) errors.push(`${label} 的 episode 必须是 1–9999 的整数。`);
            if (example.season !== undefined && (!Number.isInteger(example.season) || example.season < 1 || example.season > 99)) errors.push(`${label} 的 season 必须是 1–99 的整数或留空。`);
            if (slots.includes('season') && example.season === undefined) errors.push(`${label} 需要填写 season，因为模板包含 {season}。`);
        });

        if (new Set(examples.map((example) => example.filename).filter(Boolean)).size < MIN_EXAMPLES) errors.push('至少需要 2 个不同文件的明确标注样本。');
        if (new Set(examples.map((example) => example.episode).filter(Number.isInteger)).size < MIN_EXAMPLES) errors.push('至少需要 2 个不同集号的明确标注样本。');
        if (draft.scope === 'release' && new Set(examples.map((example) => example.title.toLocaleLowerCase()).filter(Boolean)).size < MIN_EXAMPLES) errors.push('发布组范围至少需要 2 部不同作品的样本。');
        if (filenames.length > MAX_FILENAMES) errors.push(`批量文件名最多 ${MAX_FILENAMES} 个，请删减后再预览。`);

        return {valid: errors.length === 0, error: errors[0] || '', errors, payload};
    }

    function clearPreviewOutput() {
        refs.summary.querySelectorAll('[data-summary-key]').forEach((item) => { item.textContent = '0'; });
        refs.table.replaceChildren();
        refs.empty.hidden = false;
        refs.tableWrap.hidden = true;
        renderWarnings([]);
    }

    function markDirty({clearResult = false} = {}) {
        state.inputVersion += 1;
        state.previewSerial += 1;
        state.previewBusy = false;
        state.preview = null;
        state.inputFingerprint = fingerprint(readPayload());
        refs.frame.setAttribute('aria-busy', 'false');
        clearExampleResults();
        if (clearResult) clearPreviewOutput();
        setTicket('输入已变更');
        text(refs.previewState, '输入已变更，请刷新预览');
        setMessage('输入已变更，请重新刷新预览。');
        syncButtons();
    }

    function syncButtons() {
        refs.preview.disabled = state.previewBusy || state.saveBusy;
        const ticket = state.preview;
        const canSave = Boolean(
            !state.saveBusy && ticket?.canSave && ticket.previewToken
            && ticket.inputVersion === state.inputVersion
            && ticket.fingerprint === state.inputFingerprint,
        );
        refs.save.disabled = !canSave;
    }

    function syncInputSnapshot() {
        const current = fingerprint(readPayload());
        if (current !== state.inputFingerprint) markDirty();
        return current;
    }

    function showValidation(input) {
        setMessage(input.error || '请先补全表单。', 'error');
        text(refs.previewState, '输入尚未通过校验');
        setTicket('等待修正');
        syncButtons();
    }

    function updateFilenameCount() {
        const count = readFilenames().length;
        text(refs.filenameCount, `${count} / ${MAX_FILENAMES}`);
        refs.filenameCount.classList.toggle('is-over', count > MAX_FILENAMES);
    }

    function syncScopeVisibility() {
        const release = refs.scope.value === 'release';
        refs.parentField.hidden = release;
        refs.releaseNote.hidden = !release;
        text(refs.scopeNote.querySelector('span'), release
            ? '发布组 · 跨作品无需适用文件夹；样本需覆盖不同作品，规则可跨目录复用。'
            : '仅此文件夹（不含子文件夹）：按整理起点生成的源目录名/相对目录精确匹配。');
        syncDirectoryContext();
        syncDirectoryPickerControls();
    }

    function setSummary(values = {}) {
        refs.summary.querySelectorAll('[data-summary-key]').forEach((item) => {
            const value = Number(values[item.dataset.summaryKey]);
            item.textContent = Number.isFinite(value) ? String(value) : '0';
        });
    }

    const display = (value) => value === undefined || value === null || value === '' ? '—' : String(value);

    function evidence(record, label) {
        const value = record && typeof record === 'object' ? record : {};
        const wrapper = node('div', 'release-format-preview-value');
        wrapper.append(
            node('span', '', `title: ${display(value.title)}`),
            node('small', '', `season: ${display(value.season)} · episode: ${display(value.episode)}`),
        );
        const cell = node('td');
        cell.dataset.label = label;
        cell.append(wrapper);
        return cell;
    }

    function renderPreviewRows(rows) {
        refs.table.replaceChildren();
        if (!Array.isArray(rows) || !rows.length) {
            const row = node('tr');
            const cell = node('td', '', '本批次没有可显示的结果。');
            cell.colSpan = 4;
            row.append(cell);
            refs.table.append(row);
            return;
        }
        rows.forEach((item) => {
            const status = String(item?.status || 'unmatched');
            const statusCell = node('td');
            statusCell.append(node('span', `release-format-preview-status is-${status}`, STATUS_LABELS[status] || status));
            if (item?.reason) statusCell.append(node('span', 'release-format-preview-reason', String(item.reason)));
            const row = node('tr');
            row.append(
                node('td', 'release-format-preview-filename', display(item?.filename)),
                evidence(item?.before, '原识别'), evidence(item?.after, '教学后'), statusCell,
            );
            refs.table.append(row);
        });
    }

    function renderWarnings(values) {
        const warnings = Array.isArray(values) ? values.filter(Boolean).map(String) : [];
        refs.warnings.replaceChildren();
        refs.warnings.hidden = !warnings.length;
        if (!warnings.length) return;
        refs.warnings.append(node('strong', '', '预览提示'));
        const list = node('ul');
        warnings.forEach((warning) => list.append(node('li', '', warning)));
        refs.warnings.append(list);
    }

    function renderExampleResults(values) {
        const results = Array.isArray(values) ? values : [];
        exampleRows().forEach((row, index) => {
            const result = row.querySelector('[data-example-result]');
            const passed = results[index]?.passed;
            if (!result) return;
            result.classList.remove('is-passed', 'is-failed');
            result.textContent = passed === undefined ? '已送检' : passed === true ? '已通过' : '需调整';
            if (passed !== undefined) result.classList.add(passed === true ? 'is-passed' : 'is-failed');
        });
    }

    function renderPreview(data, version, inputFingerprint) {
        setSummary(data.summary);
        renderPreviewRows(data.rows);
        renderWarnings(data.warnings);
        renderExampleResults(data.examples);
        refs.empty.hidden = true;
        refs.tableWrap.hidden = false;
        const previewToken = typeof data.preview_token === 'string' ? data.preview_token : '';
        const canSave = data.can_save === true && Boolean(previewToken);
        text(refs.previewState, canSave ? '预览通过，可保存当前输入' : '预览完成，但当前规则不能保存');
        state.preview = {canSave, previewToken, inputVersion: version, fingerprint: inputFingerprint};
        setTicket(canSave ? '票据有效 · 可保存' : '票据不可保存', canSave);
        setMessage(canSave ? '当前输入已通过预览，可保存并复用。' : '预览已完成，请根据未匹配、冲突或提示调整。', canSave ? 'success' : '');
        syncButtons();
    }

    function renderPreviewError(message) {
        text(refs.previewState, message);
        setMessage(message, 'error');
        setTicket('预览失败');
        renderWarnings([message]);
        syncButtons();
    }

    const currentRequest = (serial, version, inputFingerprint) => serial === state.previewSerial && version === state.inputVersion && inputFingerprint === state.inputFingerprint;

    async function preview() {
        syncInputSnapshot();
        const input = collectInput();
        if (!input.valid) {
            workbench.activate('editor');
            return showValidation(input);
        }
        const version = state.inputVersion;
        const inputFingerprint = fingerprint(input.payload);
        const serial = ++state.previewSerial;
        state.previewBusy = true;
        state.preview = null;
        workbench.activate('preview');
        setTicket('预览中');
        setMessage('正在刷新预览，保留上一版结果…');
        text(refs.previewState, '正在刷新，上一版结果仍保留');
        refs.frame.setAttribute('aria-busy', 'true');
        syncButtons();
        try {
            const data = await requestJson('/api/tools/release-formats/preview', {
                method: 'POST', body: JSON.stringify(input.payload),
            });
            if (!currentRequest(serial, version, inputFingerprint)) return;
            renderPreview(data, version, inputFingerprint);
        } catch (error) {
            if (currentRequest(serial, version, inputFingerprint)) renderPreviewError(error.message || '预览失败，请重试。');
        } finally {
            if (!currentRequest(serial, version, inputFingerprint)) return;
            state.previewBusy = false;
            refs.frame.setAttribute('aria-busy', 'false');
            syncButtons();
        }
    }

    function renderRuleListMessage(message, icon = 'inbox') {
        const empty = node('div', 'tmdb-regex-rule-empty release-formats-empty');
        empty.append(node('i', '', undefined), node('span', '', message));
        empty.firstChild.dataset.lucide = icon;
        refs.list.replaceChildren(empty);
        iconize(refs.list);
    }

    function renderRuleCard(rule) {
        const disabled = rule.disabled;
        const id = String(rule.id);
        const card = node('article', `tmdb-regex-rule-card release-format-rule-card${disabled ? ' is-disabled' : ''}`);
        card.dataset.ruleId = id;

        const copy = node('button', 'tmdb-regex-rule-copy release-format-rule-copy');
        copy.type = 'button';
        copy.dataset.releaseRuleSelect = id;
        copy.title = '复制为新草稿';
        copy.setAttribute('aria-label', `复制「${rule.name}」为新草稿`);
        const main = node('span', 'tmdb-regex-rule-main');
        main.append(
            node('strong', '', String(rule.name || '未命名规则')),
            node('span', `release-format-rule-state${disabled ? '' : ' is-enabled'}`, disabled ? '停用' : '启用'),
            node('span', 'tmdb-regex-priority', `v${ruleRevision(rule)}`),
        );
        copy.append(
            main,
            node('span', 'tmdb-regex-rule-meta', `${SCOPE_LABELS[rule.scope]}${rule.parent_path ? ` · ${rule.parent_path}` : ''}`),
            node('code', 'tmdb-regex-rule-pattern', String(rule.template || '')),
        );

        const actions = node('div', 'tmdb-regex-rule-actions');
        const toggle = node('button', 'icon-action');
        toggle.type = 'button';
        toggle.dataset.releaseAction = 'toggle';
        toggle.dataset.ruleId = id;
        toggle.setAttribute('aria-label', disabled ? '启用规则' : '停用规则');
        toggle.title = disabled ? '启用规则' : '停用规则';
        toggle.append(node('i', '', undefined));
        toggle.firstChild.dataset.lucide = disabled ? 'play' : 'pause';
        const remove = node('button', 'icon-action');
        remove.type = 'button';
        remove.dataset.releaseAction = 'delete';
        remove.dataset.ruleId = id;
        remove.setAttribute('aria-label', '删除规则');
        remove.title = '删除规则';
        remove.append(node('i', '', undefined));
        remove.firstChild.dataset.lucide = 'trash-2';
        actions.append(toggle, remove);
        card.append(copy, actions);
        return card;
    }

    function renderRules() {
        if (state.rules.length) {
            refs.list.replaceChildren(...state.rules.map(renderRuleCard));
            iconize(refs.list);
        } else {
            renderRuleListMessage('暂无已保存规则，可新建第一条发布格式。');
        }
        const enabled = state.rules.filter((rule) => !rule.disabled).length;
        const summary = state.rules.length ? `${state.rules.length} 条规则 · ${enabled} 条启用` : '暂无已保存规则';
        text(refs.ruleSummary, summary);
        text(refs.listState, state.rules.length ? `${state.rules.length} 条规则 · 可点击复制为新草稿` : '暂无已保存规则');
    }

    async function loadRules(background = false) {
        const serial = ++state.rulesRequestSerial;
        if (!background && !state.rulesLoaded) renderRuleListMessage('正在读取已保存规则…', 'loader-circle');
        text(refs.listState, background ? '正在刷新，保留当前列表…' : '正在读取规则…');
        try {
            const data = await requestJson('/api/tools/release-formats');
            if (serial !== state.rulesRequestSerial) return;
            if (!Array.isArray(data.items)) throw new Error('规则列表响应格式无效。');
            state.rules = data.items.map(requireRuleItem);
            state.rulesLoaded = true;
            renderRules();
        } catch (error) {
            if (serial !== state.rulesRequestSerial) return;
            text(refs.listState, error.message || '规则读取失败');
            if (!state.rules.length) renderRuleListMessage(error.message || '规则读取失败', 'circle-alert');
        }
    }

    function updateRule(item) {
        const validItem = requireRuleItem(item);
        state.rulesRequestSerial += 1;
        const index = state.rules.findIndex((rule) => rule.id === validItem.id);
        state.rules = index < 0 ? [validItem, ...state.rules] : state.rules.map((rule, i) => i === index ? validItem : rule);
        renderRules();
    }

    function resetEditor({clearResult = true} = {}) {
        invalidateDirectoryPicker();
        state.directoryContext = null;
        state.directoryStart = null;
        refs.name.value = '';
        refs.scope.value = 'directory';
        refs.parentPath.value = '';
        refs.template.value = '';
        refs.filenames.value = '';
        renderExampleRows([{}, {}]);
        text(refs.editorTitle, '新建发布格式');
        syncScopeVisibility();
        updateFilenameCount();
        markDirty({clearResult});
        setMessage('填写规则并生成预览后才能保存。');
        text(refs.previewState, '等待输入');
        setTicket('尚未预览');
    }

    function copyRuleToDraft(rule) {
        requireRuleItem(rule);
        invalidateDirectoryPicker();
        state.directoryContext = null;
        state.directoryStart = null;
        refs.name.value = rule.name;
        refs.scope.value = rule.scope;
        refs.parentPath.value = rule.parent_path;
        refs.template.value = rule.template;
        refs.filenames.value = rule.examples.map((example) => example.filename).filter(Boolean).join('\n');
        renderExampleRows(rule.examples);
        text(refs.editorTitle, `复制「${rule.name}」为新草稿`);
        syncScopeVisibility();
        updateFilenameCount();
        markDirty({clearResult: true});
        setMessage('已复制为新草稿；保存会创建新规则或复用完全相同的格式，不会修改原规则。');
        text(refs.previewState, '等待重新预览');
        setTicket('尚未预览');
        workbench.activate('editor', {resetScroll: true});
        refs.name.focus({preventScroll: true});
    }

    async function toggleRule(rule, trigger) {
        const disabled = rule.disabled;
        if (!await window.appConfirm({
            trigger,
            title: disabled ? '启用发布格式规则' : '停用发布格式规则',
            message: disabled
                ? `启用「${rule.name || '未命名规则'}」前，服务会重新回放已知教学样本；回放失败时不会启用。`
                : `停用「${rule.name || '未命名规则'}」后，文件会回到现有识别流程，不会删除规则。`,
            confirmText: disabled ? '启用规则' : '停用规则',
            danger: !disabled,
        })) return;
        try {
            const data = await requestJson(`/api/tools/release-formats/${encodeURIComponent(rule.id)}`, {
                method: 'PUT', body: JSON.stringify({disabled: !disabled, revision: ruleRevision(rule)}),
            });
            const item = requireRuleItem(data.item);
            updateRule(item);
            markDirty();
            setMessage('规则库已变化，请重新预览。');
        } catch (error) {
            text(refs.listState, error.status === 409 ? '规则版本已变化，请刷新后重试。' : (error.message || '规则启停失败'));
            if (error.status === 409) loadRules(true);
        }
    }

    async function deleteRule(rule, trigger) {
        if (!await window.appConfirm({
            trigger,
            title: '删除发布格式规则',
            message: `删除「${rule.name || '未命名规则'}」后，命名流程不会再使用它；这不会删除任何文件。`,
            confirmText: '删除规则',
            danger: true,
        })) return;
        try {
            const data = await requestJson(`/api/tools/release-formats/${encodeURIComponent(rule.id)}`, {
                method: 'DELETE', body: JSON.stringify({revision: ruleRevision(rule)}),
            });
            if (data.deleted !== true) throw new Error('删除响应格式无效。');
            state.rulesRequestSerial += 1;
            state.rules = state.rules.filter((item) => item.id !== rule.id);
            renderRules();
            markDirty();
            setMessage('规则库已变化，请重新预览。');
        } catch (error) {
            text(refs.listState, error.status === 409 ? '规则版本已变化，请刷新后重试。' : (error.message || '规则删除失败'));
            if (error.status === 409) loadRules(true);
        }
    }

    async function save(event) {
        event.preventDefault();
        if (state.saveBusy) return;
        syncInputSnapshot();
        const input = collectInput();
        if (!input.valid) return showValidation(input);
        const inputFingerprint = fingerprint(input.payload);
        const ticket = state.preview;
        const validTicket = Boolean(
            ticket?.canSave && ticket.previewToken
            && ticket.inputVersion === state.inputVersion
            && ticket.fingerprint === inputFingerprint
            && inputFingerprint === state.inputFingerprint,
        );
        if (!validTicket) {
            setMessage('当前输入没有有效预览票据，请重新刷新预览。', 'error');
            text(refs.previewState, '保存已锁定，需重新预览');
            setTicket('票据无效');
            syncButtons();
            return;
        }

        const version = state.inputVersion;
        state.saveBusy = true;
        setMessage('正在保存当前规则…');
        syncButtons();
        try {
            const data = await requestJson('/api/tools/release-formats', {
                method: 'POST',
                body: JSON.stringify({...input.payload, preview_token: ticket.previewToken, confirmed: true}),
            });
            if (typeof data.created !== 'boolean') throw new Error('保存响应缺少 created。');
            const item = requireRuleItem(data.item);
            updateRule(item);
            loadRules(true);
            if (!currentRequest(state.previewSerial, version, inputFingerprint)) return;
            state.preview = null;
            setMessage(data.created ? '规则已保存，可从列表复制为新草稿。' : '已有相同格式，未重复创建', 'success');
            text(refs.previewState, '保存完成；再次保存前需重新预览');
            setTicket('已保存');
        } catch (error) {
            if (!currentRequest(state.previewSerial, version, inputFingerprint)) return;
            if (error.status === 409) {
                state.preview = null;
                setMessage('预览已过期或出现回退，请重新刷新预览。', 'error');
                text(refs.previewState, '保存被拒绝，需要重新预览');
                setTicket('票据过期');
            } else setMessage(error.message || '规则保存失败。', 'error');
        } finally {
            state.saveBusy = false;
            syncButtons();
        }
    }

    function rememberSelection() {
        const start = Number(refs.template.selectionStart);
        const end = Number(refs.template.selectionEnd);
        if (Number.isInteger(start) && Number.isInteger(end)) state.templateSelection = {start, end};
    }

    function insertField(field) {
        const token = TOKENS[field];
        if (!token) return;
        const value = refs.template.value;
        const start = Math.max(0, Math.min(state.templateSelection.start || 0, value.length));
        const end = Math.max(start, Math.min(state.templateSelection.end || start, value.length));
        refs.template.value = `${value.slice(0, start)}${token}${value.slice(end)}`;
        const caret = start + token.length;
        refs.template.focus({preventScroll: true});
        refs.template.setSelectionRange(caret, caret);
        state.templateSelection = {start: caret, end: caret};
        refs.template.dispatchEvent(new Event('input', {bubbles: true}));
    }

    refs.form.addEventListener('input', (event) => {
        if (!state.directorySelectionApplying) {
            invalidateDirectoryPicker();
            if (event.target === refs.parentPath) state.directoryContext = null;
        }
        syncDirectoryContext();
        if (event.target === refs.scope) syncScopeVisibility();
        updateFilenameCount();
        markDirty();
    });
    refs.form.addEventListener('change', (event) => {
        if (!state.directorySelectionApplying) invalidateDirectoryPicker();
        if (event.target === refs.scope) syncScopeVisibility();
        updateFilenameCount();
        markDirty();
    });
    refs.directoryPick.addEventListener('click', () => pickDirectory('folder'));
    refs.directoryPickStart.addEventListener('click', () => pickDirectory('start'));
    refs.directoryOrigin.addEventListener('change', () => {
        invalidateDirectoryPicker();
        state.directoryStart = null;
        state.directoryContext = null;
        refreshDirectorySources();
    });
    refs.directorySource.addEventListener('change', () => {
        invalidateDirectoryPicker();
        state.directoryStart = null;
        state.directoryContext = null;
        syncDirectoryContext();
    });
    ['select', 'keyup', 'mouseup', 'focus', 'blur'].forEach((event) => refs.template.addEventListener(event, rememberSelection));
    modal.querySelectorAll('[data-release-field]').forEach((button) => {
        button.addEventListener('mousedown', (event) => event.preventDefault());
        button.addEventListener('click', () => insertField(button.dataset.releaseField));
    });

    refs.addExample.addEventListener('click', () => {
        if (exampleRows().length >= MAX_EXAMPLES) return;
        refs.examples.append(buildExampleRow());
        syncExampleControls();
        iconize(refs.examples);
        markDirty();
    });
    refs.loadExample.addEventListener('click', () => {
        invalidateDirectoryPicker();
        state.directoryContext = null;
        state.directoryStart = null;
        refs.name.value = TEACHING_EXAMPLE.name;
        refs.scope.value = TEACHING_EXAMPLE.scope;
        refs.parentPath.value = TEACHING_EXAMPLE.parent_path;
        refs.template.value = TEACHING_EXAMPLE.template;
        refs.filenames.value = TEACHING_EXAMPLE.filenames.join('\n');
        renderExampleRows(TEACHING_EXAMPLE.examples);
        syncScopeVisibility();
        updateFilenameCount();
        markDirty({clearResult: true});
        setMessage('已载入真实教学示例，点击刷新预览查看字段纠偏。');
        text(refs.previewState, '教学示例已载入');
        refs.name.focus({preventScroll: true});
    });
    refs.examples.addEventListener('click', (event) => {
        const remove = event.target.closest('[data-remove-example]');
        if (!remove || remove.disabled || exampleRows().length <= MIN_EXAMPLES) return;
        remove.closest('[data-example-row]')?.remove();
        syncExampleControls();
        markDirty();
    });
    refs.list.addEventListener('click', (event) => {
        const select = event.target.closest('[data-release-rule-select]');
        if (select) {
            const rule = state.rules.find((item) => String(item.id) === String(select.dataset.releaseRuleSelect));
            if (rule) copyRuleToDraft(rule);
            return;
        }
        const action = event.target.closest('[data-release-action]');
        if (!action) return;
        const rule = state.rules.find((item) => String(item.id) === String(action.dataset.ruleId));
        if (!rule) return;
        if (action.dataset.releaseAction === 'toggle') toggleRule(rule, action);
        if (action.dataset.releaseAction === 'delete') deleteRule(rule, action);
    });

    openButton.addEventListener('click', (event) => {
        workbench.activate('ledger', {resetScroll: true});
        lifecycle.open(event.currentTarget, {initialFocus: workbench.isMobile() ? '#releaseFormatsLedgerTab' : '#releaseFormatName'});
        if (!state.rulesLoaded) loadRules();
        refreshDirectorySources();
    });
    $('newReleaseFormatBtn').addEventListener('click', () => {
        resetEditor();
        workbench.activate('editor', {resetScroll: true});
        refs.name.focus({preventScroll: true});
    });
    refs.preview.addEventListener('click', preview);
    refs.form.addEventListener('submit', save);

    syncScopeVisibility();
    updateFilenameCount();
    syncExampleControls();
    setSummary();
    state.inputFingerprint = fingerprint(readPayload());
    iconize(modal);
})();
