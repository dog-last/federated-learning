import type { DashboardState, RuntimeLogRecord } from './dashboard';

const LEVEL_COLORS: Record<string, string> = {
    DEBUG: '#8b949e',
    INFO: '#58a6ff',
    CRITICAL: '#d29922',
    ERROR: '#ff7b72',
};

let logEl: HTMLElement | null = null;
let levelFilter: HTMLSelectElement | null = null;
let categoryFilter: HTMLSelectElement | null = null;
let sourceFilter: HTMLInputElement | null = null;
let regexFilter: HTMLInputElement | null = null;
let allLogs: RuntimeLogRecord[] = [];
let maxEvents = 1000;
let renderedCount = 0;
let activeFilterKey = '';
let pendingLogs: RuntimeLogRecord[] = [];
let flushScheduled = false;

export function initEventLog(state: DashboardState) {
    logEl = document.getElementById('event-log')!;
    levelFilter = document.getElementById('log-level-filter') as HTMLSelectElement | null;
    categoryFilter = document.getElementById('log-category-filter') as HTMLSelectElement | null;
    sourceFilter = document.getElementById('log-source-filter') as HTMLInputElement | null;
    regexFilter = document.getElementById('log-regex-filter') as HTMLInputElement | null;
    if (!logEl) return;

    bindFilters();
    if (levelFilter && !levelFilter.value) levelFilter.value = 'INFO';
    setRuntimeLogs(state.runtimeLogs);

    state.onUpdate.push(() => {
        setRuntimeLogs(state.runtimeLogs);
    });
}

export function setRuntimeLogs(records: RuntimeLogRecord[]) {
    const nextLogs = records.slice(-maxEvents);
    const nextFilterKey = getFilterKey();
    const canAppend =
        logEl &&
        nextFilterKey === activeFilterKey &&
        nextLogs.length >= allLogs.length &&
        allLogs.every((record, idx) => record === nextLogs[idx] || record.line === nextLogs[idx]?.line);

    if (canAppend) {
        const incoming = nextLogs.slice(allLogs.length);
        allLogs = nextLogs;
        appendVisibleLogs(incoming);
        trimRenderedLogs();
        return;
    }

    allLogs = nextLogs;
    renderLogs();
}

export function appendRuntimeLog(record: RuntimeLogRecord) {
    allLogs.push(record);
    while (allLogs.length > maxEvents) allLogs.shift();
    pendingLogs.push(record);
    scheduleFlush();
}

function scheduleFlush() {
    if (flushScheduled) return;
    flushScheduled = true;
    requestAnimationFrame(() => {
        flushScheduled = false;
        const batch = pendingLogs.splice(0, pendingLogs.length);
        appendVisibleLogs(batch);
        trimRenderedLogs();
    });
}

function bindFilters() {
    for (const el of [levelFilter, categoryFilter, sourceFilter, regexFilter]) {
        el?.addEventListener('input', renderLogs);
        el?.addEventListener('change', renderLogs);
    }
}

function renderLogs() {
    if (!logEl) return;
    const shouldStickToBottom = isNearBottom(logEl);
    const previousScrollLeft = logEl.scrollLeft;
    activeFilterKey = getFilterKey();
    logEl.replaceChildren();
    const filtered = allLogs.filter(matchesFilters);
    for (const record of filtered) {
        logEl.appendChild(renderLine(record, false));
    }
    renderedCount = filtered.length;
    logEl.scrollLeft = previousScrollLeft;
    if (shouldStickToBottom) logEl.scrollTop = logEl.scrollHeight;
}

function appendVisibleLogs(records: RuntimeLogRecord[]) {
    if (!logEl || !records.length) return;
    const shouldStickToBottom = isNearBottom(logEl);
    const previousScrollLeft = logEl.scrollLeft;
    activeFilterKey = getFilterKey();
    for (const record of records) {
        if (!matchesFilters(record)) continue;
        logEl.appendChild(renderLine(record, true));
        renderedCount++;
    }
    logEl.scrollLeft = previousScrollLeft;
    if (shouldStickToBottom) logEl.scrollTop = logEl.scrollHeight;
}

function trimRenderedLogs() {
    if (!logEl) return;
    while (renderedCount > maxEvents && logEl.firstElementChild) {
        logEl.firstElementChild.remove();
        renderedCount--;
    }
}

function isNearBottom(el: HTMLElement): boolean {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 24;
}

function getFilterKey(): string {
    return JSON.stringify({
        level: levelFilter?.value || '',
        category: categoryFilter?.value || '',
        source: sourceFilter?.value || '',
        regex: regexFilter?.value || '',
    });
}

function matchesFilters(record: RuntimeLogRecord): boolean {
    const level = levelFilter?.value || '';
    const category = categoryFilter?.value || '';
    const source = (sourceFilter?.value || '').trim().toLowerCase();
    const pattern = (regexFilter?.value || '').trim();

    if (level && record.level !== level) return false;
    if (category && record.category !== category) return false;
    if (source && !String(record.source || '').toLowerCase().includes(source)) return false;
    if (!pattern) return true;

    try {
        return new RegExp(pattern, 'i').test(record.line);
    } catch {
        regexFilter?.classList.add('invalid');
        return true;
    } finally {
        if (pattern) {
            try {
                new RegExp(pattern);
                regexFilter?.classList.remove('invalid');
            } catch {
                // Keep invalid marker from the catch branch above.
            }
        } else {
            regexFilter?.classList.remove('invalid');
        }
    }
}

function renderLine(record: RuntimeLogRecord, animate: boolean): HTMLElement {
    const div = document.createElement('div');
    div.className = `event-line level-${record.level.toLowerCase()} category-${record.category}`;
    if (animate) div.classList.add('event-line-new');
    div.title = record.line;

    const level = document.createElement('span');
    level.className = 'log-level';
    level.textContent = record.level;
    level.style.color = LEVEL_COLORS[record.level] || '#c9d1d9';

    const category = document.createElement('span');
    category.className = 'log-category';
    category.textContent = record.category;

    const body = document.createElement('span');
    body.className = 'log-body';
    body.textContent = record.line;

    div.append(level, category, body);
    return div;
}
