import type { DashboardState, ClientState } from './dashboard';

const STATUS_COLORS: Record<string, string> = {
    idle: '#8b949e', online: '#58a6ff', training: '#3fb950',
    sending: '#f0883e', receiving: '#d29922', done: '#39d353',
    disconnected: '#ff7b72', waiting: '#8b949e', ready: '#79c0ff',
    scheduled: '#bc8cff', trained: '#3fb950',
};

function fmtBytes(bytes: number): string {
    if (bytes < 1024) return bytes + 'B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + 'KB';
    return (bytes / 1048576).toFixed(1) + 'MB';
}

export function initClientTable(state: DashboardState) {
    state.onUpdate.push(() => renderClientTable(state));
}

function renderClientTable(state: DashboardState) {
    const tbody = document.getElementById('client-tbody');
    if (!tbody) return;

    const clientIds = Object.keys(state.clients);
    if (clientIds.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty">Waiting for clients...</td></tr>';
        return;
    }

    let html = '';
    for (const cid of clientIds) {
        const c = state.clients[cid];
        const statusColor = STATUS_COLORS[c.status] || '#8b949e';
        const progressPct = Math.round((c.progress || 0) * 100);
        const loss = c.train_loss != null ? c.train_loss.toFixed(4) : '-';
        const acc = c.train_acc != null ? (c.train_acc * 100).toFixed(1) + '%' : '-';
        const net = state.sourceNetTotals[cid] || { bytes_sent: 0, bytes_recv: 0 };
        const netText = `tx=${fmtBytes(net.bytes_sent)} rx=${fmtBytes(net.bytes_recv)}`;

        html += `<tr>
            <td class="client-id">${cid}</td>
            <td><span class="status-badge" style="background:${statusColor}20;color:${statusColor}">${c.status}</span></td>
            <td><div class="progress-bar"><div class="progress-fill" style="width:${progressPct}%;background:${statusColor}"></div></div><span class="progress-text">${progressPct}%</span></td>
            <td class="mono">${loss}</td>
            <td class="mono">${acc}</td>
            <td class="mono">${netText}</td>
        </tr>`;
    }
    tbody.innerHTML = html;
}
