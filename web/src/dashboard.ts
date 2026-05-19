import { appendEvent } from './event-log';

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
                    state.metrics = msg.data;
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
        const rd = event.round;
        if (rd && !state.metrics.rounds.includes(rd)) {
            state.metrics.rounds.push(rd);
            state.metrics.train_loss.push(event.train_loss ?? null);
            state.metrics.train_acc.push(event.train_acc ?? null);
            state.metrics.val_loss.push(event.val_loss ?? null);
            state.metrics.val_acc.push(event.val_acc ?? null);
            state.metrics.test_loss.push(event.test_loss ?? null);
            state.metrics.test_acc.push(event.test_acc ?? null);
        }
        state.phase = 'round complete';
    }
    if (eventType === 'ring_global_eval') {
        // ring_global_eval contains the actual global metrics for ring mode
        const rd = event.round;
        if (rd && !state.metrics.rounds.includes(rd)) {
            state.metrics.rounds.push(rd);
            state.metrics.train_loss.push(event.train_loss ?? null);
            state.metrics.train_acc.push(event.train_acc ?? null);
            state.metrics.val_loss.push(event.val_loss ?? null);
            state.metrics.val_acc.push(event.val_acc ?? null);
            state.metrics.test_loss.push(event.test_loss ?? null);
            state.metrics.test_acc.push(event.test_acc ?? null);
        }
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

            // For ring mode: store per-client metrics history for visualization
            if (eventType === 'ring_local_train_done' && event.node_id != null && event.round != null) {
                const round = event.round;
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
                const pc = state.metrics.per_client[cid];
                // Only add if this round hasn't been recorded for this client
                if (!pc.rounds.includes(round)) {
                    pc.rounds.push(round);
                    pc.train_loss.push(event.train_loss ?? null);
                    pc.train_acc.push(event.train_acc ?? null);
                    pc.val_loss.push(event.val_loss ?? null);
                    pc.val_acc.push(event.val_acc ?? null);
                    pc.test_loss.push(event.test_loss ?? null);
                    pc.test_acc.push(event.test_acc ?? null);
                }
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
    if (logLine) {
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        state.keyEvents.push(`[${ts}] ${logLine}`);
        appendEvent(`[${ts}] ${logLine}`);
    }
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
