import { appendRuntimeLog, setRuntimeLogs } from './event-log';

export interface DashboardState {
    ws: WebSocket | null;
    connected: boolean;
    phase: string;
    mode: string;
    currentRound: number;
    epochTotal: number;
    clients: Record<string, ClientState>;
    keyEvents: string[];
    metrics: MetricsHistory;
    topologyMode: string;
    onUpdate: Array<() => void>;
    sourceNetTotals: Record<string, { bytes_sent: number; bytes_recv: number }>;
    runtimeLogs: RuntimeLogRecord[];
}

export interface RuntimeLogRecord {
    ts: number;
    line: string;
    level: string;
    category: string;
    source: string;
    event_type: string;
    round?: number | string | null;
    peer?: string | null;
    direction?: string | null;
    message_type?: string | null;
    payload_label?: string | null;
    payload_bytes?: number | null;
    bytes_sent_total?: number | null;
    bytes_recv_total?: number | null;
    messages_sent_total?: number | null;
    messages_recv_total?: number | null;
    reason?: string | null;
    error?: string | null;
    status?: string | null;
}

export interface ClientState {
    status: string;
    progress: number;
    train_loss: number | null;
    train_acc: number | null;
    test_acc: number | null;
    local_epoch: string;
}

export interface MetricsHistory {
    rounds: number[];
    train_loss: (number | null)[];
    train_acc: (number | null)[];
    val_loss: (number | null)[];
    val_acc: (number | null)[];
    test_loss: (number | null)[];
    test_acc: (number | null)[];
    per_client: Record<string, {
        rounds: number[];
        train_loss: (number | null)[];
        train_acc: (number | null)[];
        val_loss: (number | null)[];
        val_acc: (number | null)[];
        test_loss: (number | null)[];
        test_acc: (number | null)[];
    }>;
}

export function initDashboard(): DashboardState {
    const state: DashboardState = {
        ws: null,
        connected: false,
        phase: 'idle',
        mode: '-',
        currentRound: 0,
        epochTotal: 0,
        clients: {},
        keyEvents: [],
        metrics: { rounds: [], train_loss: [], train_acc: [], val_loss: [], val_acc: [], test_loss: [], test_acc: [], per_client: {} },
        topologyMode: 'centralized',
        onUpdate: [],
        sourceNetTotals: {},
        runtimeLogs: [],
    };
    connectWebSocket(state);
    return state;
}

function normalizeSourceId(source: string, mode: string): string {
    if (!source) return source;
    if (mode === 'ring') {
        const match = /^ring_node_(\d+)$/.exec(source);
        if (match) return `client_${match[1]}`;
    }
    return source;
}

function connectWebSocket(state: DashboardState) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;
    let retryDelay = 1000;

    console.log('[WebSocket] Connecting to:', wsUrl);

    function connect() {
        updateConnectionStatus(state, 'connecting');
        console.log('[WebSocket] Attempting connection...');
        const ws = new WebSocket(wsUrl);
        state.ws = ws;

        ws.onopen = () => {
            state.connected = true;
            retryDelay = 1000;
            console.log('[WebSocket] Connected successfully');
            updateConnectionStatus(state, 'connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'state_snapshot') {
                    applyStateSnapshot(state, msg.data);
                } else if (msg.type === 'metrics_history') {
                    state.metrics = mergeMetrics(state.metrics, normalizeMetrics(msg.data));
                    refreshDerivedMetrics(state);
                } else if (msg.type === 'runtime_logs') {
                    state.runtimeLogs = msg.data || [];
                    setRuntimeLogs(state.runtimeLogs);
                } else if (msg.type === 'runtime_log') {
                    const record = msg.data as RuntimeLogRecord;
                    state.runtimeLogs.push(record);
                    if (state.runtimeLogs.length > 1000) state.runtimeLogs.shift();
                    appendRuntimeLog(record);
                } else {
                    handleEvent(state, msg);
                }
                for (const cb of state.onUpdate) cb();
            } catch (e) {
                console.error('[WebSocket] Error parsing message:', e);
            }
        };

        ws.onclose = (event) => {
            state.connected = false;
            console.log(`[WebSocket] Closed (code: ${event.code}, reason: ${event.reason || 'none'})`);
            updateConnectionStatus(state, 'disconnected');
            setTimeout(connect, retryDelay);
            retryDelay = Math.min(retryDelay * 2, 30000);
        };

        ws.onerror = (error) => {
            console.error('[WebSocket] Error:', error);
            ws.close();
        };
    }

    connect();
}

function updateConnectionStatus(state: DashboardState, status: string) {
    const el = document.getElementById('connection-status');
    if (!el) return;
    const dot = el.querySelector('.dot') as HTMLElement;
    const text = el.querySelector('.text') as HTMLElement;
    const colors: Record<string, string> = { connected: '#3fb950', connecting: '#d29922', disconnected: '#ff7b72' };
    const labels: Record<string, string> = { connected: 'Connected', connecting: 'Reconnecting...', disconnected: 'Disconnected' };
    if (dot) dot.style.backgroundColor = colors[status] || '#8b949e';
    if (text) text.textContent = labels[status] || status;
}

function applyStateSnapshot(state: DashboardState, data: any) {
    state.phase = data.phase || 'idle';
    state.mode = data.mode || '-';
    state.currentRound = data.current_round || 0;
    state.epochTotal = data.epoch_total || 0;
    state.clients = data.clients || {};
    state.keyEvents = data.key_events || [];
    state.topologyMode = data.mode || 'centralized';
    state.sourceNetTotals = {};
    const rawTotals = data.source_net_totals || {};
    for (const [source, totals] of Object.entries(rawTotals)) {
        state.sourceNetTotals[normalizeSourceId(source, data.mode || state.mode)] = totals as { bytes_sent: number; bytes_recv: number };
    }
    updateHeaderUI(state);
}

function normalizeMetricArray(values: any[], rounds: number[]): (number | null)[] {
    const out = Array.isArray(values) ? values.slice(0, rounds.length) : [];
    while (out.length < rounds.length) out.push(null);
    return out.map((value) => value === undefined ? null : value);
}

function normalizeMetrics(data: any): MetricsHistory {
    const rounds = Array.isArray(data?.rounds) ? data.rounds : [];
    return {
        rounds,
        train_loss: normalizeMetricArray(data?.train_loss, rounds),
        train_acc: normalizeMetricArray(data?.train_acc, rounds),
        val_loss: normalizeMetricArray(data?.val_loss, rounds),
        val_acc: normalizeMetricArray(data?.val_acc, rounds),
        test_loss: normalizeMetricArray(data?.test_loss, rounds),
        test_acc: normalizeMetricArray(data?.test_acc, rounds),
        per_client: data?.per_client || {},
    };
}

function mergeMetricArray(current: (number | null)[], incoming: (number | null)[], rounds: number[]): (number | null)[] {
    const out = normalizeMetricArray(current, rounds);
    const inc = normalizeMetricArray(incoming, rounds);
    return out.map((value, idx) => inc[idx] ?? value ?? null);
}

function mergeMetrics(current: MetricsHistory, incoming: MetricsHistory): MetricsHistory {
    const mergedPerClient = { ...current.per_client, ...incoming.per_client };
    const rounds = Array.from(new Set([
        ...current.rounds,
        ...incoming.rounds,
        ...roundsFromPerClient(mergedPerClient),
    ])).sort((a, b) => a - b);
    const mapByRound = (sourceRounds: number[], values: (number | null)[]) => {
        const map = new Map<number, number | null>();
        sourceRounds.forEach((round, idx) => map.set(round, values[idx] ?? null));
        return rounds.map((round) => map.get(round) ?? null);
    };
    const currentAligned: MetricsHistory = {
        rounds,
        train_loss: mapByRound(current.rounds, current.train_loss),
        train_acc: mapByRound(current.rounds, current.train_acc),
        val_loss: mapByRound(current.rounds, current.val_loss),
        val_acc: mapByRound(current.rounds, current.val_acc),
        test_loss: mapByRound(current.rounds, current.test_loss),
        test_acc: mapByRound(current.rounds, current.test_acc),
        per_client: mergedPerClient,
    };
    const incomingAligned: MetricsHistory = {
        rounds,
        train_loss: mapByRound(incoming.rounds, incoming.train_loss),
        train_acc: mapByRound(incoming.rounds, incoming.train_acc),
        val_loss: mapByRound(incoming.rounds, incoming.val_loss),
        val_acc: mapByRound(incoming.rounds, incoming.val_acc),
        test_loss: mapByRound(incoming.rounds, incoming.test_loss),
        test_acc: mapByRound(incoming.rounds, incoming.test_acc),
        per_client: mergedPerClient,
    };
    return {
        rounds,
        train_loss: mergeMetricArray(currentAligned.train_loss, incomingAligned.train_loss, rounds),
        train_acc: mergeMetricArray(currentAligned.train_acc, incomingAligned.train_acc, rounds),
        val_loss: mergeMetricArray(currentAligned.val_loss, incomingAligned.val_loss, rounds),
        val_acc: mergeMetricArray(currentAligned.val_acc, incomingAligned.val_acc, rounds),
        test_loss: mergeMetricArray(currentAligned.test_loss, incomingAligned.test_loss, rounds),
        test_acc: mergeMetricArray(currentAligned.test_acc, incomingAligned.test_acc, rounds),
        per_client: mergedPerClient,
    };
}

function averagePerClient(perClient: Record<string, any>, metricKey: string, rounds: number[]): (number | null)[] {
    return rounds.map((round) => {
        let total = 0;
        let count = 0;
        for (const pc of Object.values(perClient)) {
            const idx = Array.isArray(pc.rounds) ? pc.rounds.indexOf(round) : -1;
            const value = idx >= 0 ? pc[metricKey]?.[idx] : null;
            if (typeof value === 'number') {
                total += value;
                count++;
            }
        }
        return count ? total / count : null;
    });
}

function roundsFromPerClient(perClient: Record<string, any>): number[] {
    const rounds = new Set<number>();
    for (const pc of Object.values(perClient)) {
        if (!Array.isArray(pc.rounds)) continue;
        for (const round of pc.rounds) {
            if (typeof round === 'number') rounds.add(round);
        }
    }
    return Array.from(rounds).sort((a, b) => a - b);
}

function alignMetricToRounds(sourceRounds: number[], values: (number | null)[], targetRounds: number[]): (number | null)[] {
    const byRound = new Map<number, number | null>();
    sourceRounds.forEach((round, idx) => byRound.set(round, values[idx] ?? null));
    return targetRounds.map((round) => byRound.get(round) ?? null);
}

function fillMissingWithAverage(values: (number | null)[], perClient: Record<string, any>, metricKey: string, rounds: number[]) {
    const localAverage = averagePerClient(perClient, metricKey, rounds);
    return values.map((value, idx) => value ?? localAverage[idx]);
}

function refreshDerivedMetrics(state: DashboardState) {
    const sourceRounds = state.metrics.rounds;
    const perClient = state.metrics.per_client;
    const rounds = Array.from(new Set([...sourceRounds, ...roundsFromPerClient(perClient)])).sort((a, b) => a - b);
    state.metrics.rounds = rounds;
    state.metrics.train_loss = alignMetricToRounds(sourceRounds, state.metrics.train_loss, rounds);
    state.metrics.train_acc = alignMetricToRounds(sourceRounds, state.metrics.train_acc, rounds);
    state.metrics.val_loss = alignMetricToRounds(sourceRounds, state.metrics.val_loss, rounds);
    state.metrics.val_acc = alignMetricToRounds(sourceRounds, state.metrics.val_acc, rounds);
    state.metrics.test_loss = alignMetricToRounds(sourceRounds, state.metrics.test_loss, rounds);
    state.metrics.test_acc = alignMetricToRounds(sourceRounds, state.metrics.test_acc, rounds);
    if (!rounds.length || !Object.keys(perClient).length) return;
    state.metrics.train_loss = fillMissingWithAverage(state.metrics.train_loss, perClient, 'train_loss', rounds);
    state.metrics.train_acc = fillMissingWithAverage(state.metrics.train_acc, perClient, 'train_acc', rounds);
    state.metrics.val_loss = fillMissingWithAverage(state.metrics.val_loss, perClient, 'val_loss', rounds);
    state.metrics.val_acc = fillMissingWithAverage(state.metrics.val_acc, perClient, 'val_acc', rounds);
    state.metrics.test_loss = fillMissingWithAverage(state.metrics.test_loss, perClient, 'test_loss', rounds);
    state.metrics.test_acc = fillMissingWithAverage(state.metrics.test_acc, perClient, 'test_acc', rounds);
}

function ensurePerClientMetrics(state: DashboardState, cid: string) {
    if (!state.metrics.per_client[cid]) {
        state.metrics.per_client[cid] = {
            rounds: [],
            train_loss: [],
            train_acc: [],
            val_loss: [],
            val_acc: [],
            test_loss: [],
            test_acc: [],
        };
    }
    return state.metrics.per_client[cid];
}

function recordLocalMetrics(state: DashboardState, cid: string, event: any) {
    const round = event.round;
    if (!cid || typeof round !== 'number') return;
    const pc = ensurePerClientMetrics(state, cid);
    const existing = pc.rounds.indexOf(round);
    const idx = existing === -1 ? pc.rounds.length : existing;
    if (existing === -1) pc.rounds.push(round);
    pc.train_loss[idx] = event.train_loss ?? pc.train_loss[idx] ?? null;
    pc.train_acc[idx] = event.train_acc ?? pc.train_acc[idx] ?? null;
    pc.val_loss[idx] = event.val_loss ?? pc.val_loss[idx] ?? null;
    pc.val_acc[idx] = event.val_acc ?? pc.val_acc[idx] ?? null;
    pc.test_loss[idx] = event.test_loss ?? pc.test_loss[idx] ?? null;
    pc.test_acc[idx] = event.test_acc ?? pc.test_acc[idx] ?? null;
    refreshDerivedMetrics(state);
}

function upsertMetricRound(state: DashboardState, event: any, includeTrainVal: boolean) {
    const rd = event.round;
    if (!rd) return;
    let idx = state.metrics.rounds.indexOf(rd);
    if (idx === -1) {
        state.metrics.rounds.push(rd);
        idx = state.metrics.rounds.length - 1;
        state.metrics.train_loss.push(null);
        state.metrics.train_acc.push(null);
        state.metrics.val_loss.push(null);
        state.metrics.val_acc.push(null);
        state.metrics.test_loss.push(null);
        state.metrics.test_acc.push(null);
    }
    if (includeTrainVal) {
        state.metrics.train_loss[idx] = event.train_loss ?? state.metrics.train_loss[idx] ?? null;
        state.metrics.train_acc[idx] = event.train_acc ?? state.metrics.train_acc[idx] ?? null;
        state.metrics.val_loss[idx] = event.val_loss ?? state.metrics.val_loss[idx] ?? null;
        state.metrics.val_acc[idx] = event.val_acc ?? state.metrics.val_acc[idx] ?? null;
    }
    state.metrics.test_loss[idx] = event.test_loss ?? state.metrics.test_loss[idx] ?? null;
    state.metrics.test_acc[idx] = event.test_acc ?? state.metrics.test_acc[idx] ?? null;
    refreshDerivedMetrics(state);
}

function handleEvent(state: DashboardState, event: any) {
    const eventType = event.event_type || '';

    if (eventType === 'manager_start') {
        state.mode = event.experiment?.mode || state.mode;
        state.topologyMode = event.experiment?.mode || 'centralized';
        state.epochTotal = event.experiment?.global_epochs || state.epochTotal;
    }
    if (eventType === 'round_start' || eventType === 'ring_round_start') {
        state.currentRound = event.round || state.currentRound;
        state.epochTotal = event.total_epochs || state.epochTotal;
        state.phase = 'broadcast';
    }
    // For ring mode: only ring_global_eval has the global metrics
    // For client-server mode: round_end has the metrics
    if (eventType === 'round_end') {
        state.currentRound = event.round || state.currentRound;
        upsertMetricRound(state, event, true);
        state.phase = 'round complete';
    }
    if (eventType === 'ring_global_eval') {
        upsertMetricRound(state, event, false);
        state.currentRound = event.round || state.currentRound;
        state.phase = 'global evaluation';
    }
    if (eventType === 'ring_round_end') {
        // ring_round_end marks round completion but doesn't have metrics
        state.currentRound = event.round || state.currentRound;
        state.phase = 'round complete';
    }
    if (eventType === 'batch_progress' || eventType === 'local_round_done' || eventType === 'ring_local_train_done') {
        // For ring mode events, use node_id to construct client_id (client_{node_id})
        // For client-server mode, use client_id or source
        let cid: string;
        if (eventType === 'ring_local_train_done' && event.node_id != null) {
            cid = `client_${event.node_id}`;
        } else {
            cid = event.client_id || event.source || '';
        }
        if (cid) {
            state.clients[cid] = {
                status: eventType === 'batch_progress' ? 'training' : eventType === 'ring_local_train_done' ? 'trained' : 'done',
                progress: eventType === 'batch_progress' ? (event.batch_idx || 0) / Math.max(event.total_batches || 1, 1) : 1.0,
                train_loss: event.train_loss ?? event.batch_loss ?? null,
                train_acc: event.train_acc ?? event.batch_acc ?? null,
                test_acc: event.test_acc ?? null,
                local_epoch: eventType === 'batch_progress' ? `${event.local_epoch_idx || 0}/${event.local_epochs || 1}` : '-',
            };

            if ((eventType === 'local_round_done' || eventType === 'ring_local_train_done') && event.round != null) {
                recordLocalMetrics(state, cid, event);
            }
        }
    }
    if (eventType === 'network_io' || eventType === 'send_ack' || eventType === 'recv_ack') {
        const source = normalizeSourceId(event.source || '', state.mode);
        if (source) {
            const bucket = state.sourceNetTotals[source] || { bytes_sent: 0, bytes_recv: 0 };
            if (typeof event.bytes_sent_total === 'number') bucket.bytes_sent = event.bytes_sent_total;
            if (typeof event.bytes_recv_total === 'number') bucket.bytes_recv = event.bytes_recv_total;
            state.sourceNetTotals[source] = bucket;
        }
    }

    updateHeaderUI(state);

    const logLine = formatKeyEvent(event);
    if (logLine) state.keyEvents.push(logLine);
}

function formatKeyEvent(event: any): string | null {
    const eventType = event.event_type || '';
    switch (eventType) {
        case 'round_start':
            return `[ROUND] start round=${event.round || '-'}/${event.total_epochs || '-'} mode=${event.mode || '-'} clients=${event.expected_clients || '-'}`;
        case 'round_end':
            return `[ROUND] done round=${event.round || '-'} loss=${fmt(event.test_loss)} acc=${fmt(event.test_acc)}`;
        case 'local_round_done':
            return `[CLIENT] ${event.client_id || event.source || '-'} done loss=${fmt(event.train_loss)} acc=${fmt(event.train_acc)}`;
        case 'ring_round_start':
            return `[RING] round ${event.round || '-'} start (node=${event.node_id || '-'})`;
        case 'ring_local_train_done':
            return `[RING] node ${event.node_id || '-'} trained loss=${fmt(event.train_loss)} acc=${fmt(event.train_acc)}`;
        case 'ring_global_eval':
            return `[RING] round ${event.round || '-'} eval loss=${fmt(event.test_loss)} acc=${fmt(event.test_acc)}`;
        case 'ring_round_end':
            // ring_round_end doesn't have metrics, they come from ring_global_eval
            return `[RING] round ${event.round || '-'} done`;
        case 'training_started':
            return `[CTRL] training started`;
        case 'training_stopped':
            return `[CTRL] training stopped`;
        case 'shutdown':
            return `[CTRL] shutdown source=${event.source || '-'}`;
        case 'target_reached':
            return `[CTRL] target reached`;
        case 'startup':
            return `[START] ${event.client_id || event.source || '-'} online mode=${event.mode || '-'} device=${event.device || '-'}`;
        default:
            return null;
    }
}

function fmt(v: any): string {
    if (v == null) return '-';
    return typeof v === 'number' ? v.toFixed(4) : String(v);
}

function updateHeaderUI(state: DashboardState) {
    const modeBadge = document.getElementById('mode-badge');
    if (modeBadge) modeBadge.textContent = state.mode.toUpperCase();

    const roundDisplay = document.getElementById('round-display');
    if (roundDisplay) roundDisplay.textContent = `Round ${state.currentRound}/${state.epochTotal}`;
}
