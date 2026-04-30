import type { DashboardState } from './dashboard';

const EVENT_COLORS: Record<string, string> = {
    ROUND: '#58a6ff',
    CLIENT: '#3fb950',
    RING: '#f0883e',
    CTRL: '#bc8cff',
    START: '#79c0ff',
};

let logEl: HTMLElement | null = null;
let maxEvents = 200;
let eventCount = 0;

export function initEventLog(state: DashboardState) {
    logEl = document.getElementById('event-log')!;
    if (!logEl) return;

    // Render existing key events from state snapshot
    for (const evt of state.keyEvents) {
        appendEvent(evt);
    }

    state.onUpdate.push(() => {
        // New events are handled via handleEvent in dashboard.ts
        // We check if keyEvents grew
        const latestEvent = state.keyEvents[state.keyEvents.length - 1];
        if (latestEvent && !logEl?.querySelector(`[data-event="${latestEvent}"]`)) {
            appendEvent(latestEvent);
        }
    });
}

export function appendEvent(text: string) {
    if (!logEl) return;
    const div = document.createElement('div');
    div.className = 'event-line';
    div.setAttribute('data-event', text);

    // Color by tag
    const tagMatch = text.match(/\[(\w+)\]/);
    if (tagMatch) {
        const color = EVENT_COLORS[tagMatch[1]] || '#c9d1d9';
        div.innerHTML = text.replace(`[${tagMatch[1]}]`, `<span style="color:${color};font-weight:600">[${tagMatch[1]}]</span>`);
    } else {
        div.textContent = text;
    }

    logEl.appendChild(div);
    eventCount++;

    // Trim old events
    while (eventCount > maxEvents && logEl.firstChild) {
        logEl.removeChild(logEl.firstChild);
        eventCount--;
    }

    logEl.scrollTop = logEl.scrollHeight;
}
