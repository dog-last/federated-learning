import type { DashboardState, ClientState } from './dashboard';

const STATUS_COLORS: Record<string, string> = {
    idle: '#8b949e', online: '#58a6ff', training: '#3fb950',
    sending: '#f0883e', receiving: '#d29922', done: '#39d353',
    disconnected: '#ff7b72', waiting: '#8b949e', ready: '#79c0ff',
    scheduled: '#bc8cff', trained: '#3fb950',
};

interface Particle {
    x: number; y: number; tx: number; ty: number;
    progress: number; speed: number; color: string;
}

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let particles: Particle[] = [];
let animFrame: number;

export function initTopology(state: DashboardState) {
    canvas = document.getElementById('topology-canvas') as HTMLCanvasElement;
    if (!canvas) return;
    ctx = canvas.getContext('2d')!;
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    // drawTopology is called by animateParticles loop — no need for onUpdate push
    animateParticles(state);
}

function resizeCanvas() {
    const rect = canvas.parentElement!.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
}

function drawTopology(state: DashboardState) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const clientIds = Object.keys(state.clients);
    if (clientIds.length === 0) return;

    const cx = w / 2;
    const cy = h / 2;

    if (state.topologyMode === 'ring') {
        drawRingTopology(state, clientIds, cx, cy, Math.min(w, h) * 0.35);
    } else {
        drawStarTopology(state, clientIds, cx, cy, Math.min(w, h) * 0.35);
    }
}

function drawStarTopology(state: DashboardState, clientIds: string[], cx: number, cy: number, radius: number) {
    // Draw server at center
    ctx.beginPath();
    ctx.arc(cx, cy, 22, 0, Math.PI * 2);
    ctx.fillStyle = '#58a6ff';
    ctx.fill();
    ctx.strokeStyle = 'rgba(88,166,255,0.3)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Server', cx, cy);

    // Draw clients in circle around server
    const n = clientIds.length;
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        const clientState = state.clients[clientIds[i]];
        const color = STATUS_COLORS[clientState?.status || 'idle'] || '#8b949e';

        // Connection line
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = 'rgba(139,148,158,0.3)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Client node
        ctx.beginPath();
        ctx.arc(x, y, 18, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = color + '4d';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#fff';
        ctx.font = '9px sans-serif';
        ctx.fillText(clientIds[i].replace('client_', 'C'), x, y);
    }
}

function drawRingTopology(state: DashboardState, clientIds: string[], cx: number, cy: number, radius: number) {
    const n = clientIds.length;
    for (let i = 0; i < n; i++) {
        const angle1 = (Math.PI * 2 * i / n) - Math.PI / 2;
        const angle2 = (Math.PI * 2 * ((i + 1) % n) / n) - Math.PI / 2;
        const x1 = cx + radius * Math.cos(angle1);
        const y1 = cy + radius * Math.sin(angle1);
        const x2 = cx + radius * Math.cos(angle2);
        const y2 = cy + radius * Math.sin(angle2);

        // Arrow line
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = 'rgba(139,148,158,0.3)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Arrow head
        const headLen = 8;
        const a = Math.atan2(y2 - y1, x2 - x1);
        const mx = x2 - 18 * Math.cos(a);
        const my = y2 - 18 * Math.sin(a);
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx - headLen * Math.cos(a - 0.4), my - headLen * Math.sin(a - 0.4));
        ctx.moveTo(mx, my);
        ctx.lineTo(mx - headLen * Math.cos(a + 0.4), my - headLen * Math.sin(a + 0.4));
        ctx.strokeStyle = '#8b949e';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Node
        const clientState = state.clients[clientIds[i]];
        const color = STATUS_COLORS[clientState?.status || 'idle'] || '#8b949e';
        ctx.beginPath();
        ctx.arc(x1, y1, 18, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = color + '4d';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = clientIds[i].replace('client_', 'N');
        ctx.fillText(label, x1, y1);
    }
}

function animateParticles(state: DashboardState) {
    function tick() {
        // Spawn particles on network events
        const clientIds = Object.keys(state.clients);
        for (const cid of clientIds) {
            const c = state.clients[cid];
            if (c.status === 'sending' || c.status === 'receiving') {
                spawnParticle(state, cid, c.status);
            }
        }
        // Update and draw particles
        drawTopology(state);
        drawParticles();
        animFrame = requestAnimationFrame(tick);
    }
    tick();
}

function spawnParticle(state: DashboardState, clientId: string, direction: string) {
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.35;
    const clientIds = Object.keys(state.clients);
    const idx = clientIds.indexOf(clientId);
    if (idx < 0) return;

    const angle = (Math.PI * 2 * idx / clientIds.length) - Math.PI / 2;
    const nx = cx + radius * Math.cos(angle);
    const ny = cy + radius * Math.sin(angle);

    // Rate-limit: skip if too many particles for this client
    if (particles.length > 100) return;

    if (state.topologyMode === 'ring') {
        const nextIdx = (idx + 1) % clientIds.length;
        const nextAngle = (Math.PI * 2 * nextIdx / clientIds.length) - Math.PI / 2;
        const nextX = cx + radius * Math.cos(nextAngle);
        const nextY = cy + radius * Math.sin(nextAngle);
        particles.push({
            x: direction === 'sending' ? nx : nextX,
            y: direction === 'sending' ? ny : nextY,
            tx: direction === 'sending' ? nextX : nx,
            ty: direction === 'sending' ? nextY : ny,
            progress: 0,
            speed: 0.02 + Math.random() * 0.01,
            color: '#58a6ff',
        });
    } else {
        if (direction === 'sending') {
            particles.push({ x: nx, y: ny, tx: cx, ty: cy, progress: 0, speed: 0.03 + Math.random() * 0.01, color: '#f0883e' });
        } else {
            particles.push({ x: cx, y: cy, tx: nx, ty: ny, progress: 0, speed: 0.03 + Math.random() * 0.01, color: '#3fb950' });
        }
    }
}

function drawParticles() {
    const alive: Particle[] = [];
    for (const p of particles) {
        p.progress += p.speed;
        if (p.progress >= 1) continue;
        const x = p.x + (p.tx - p.x) * p.progress;
        const y = p.y + (p.ty - p.y) * p.progress;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 1 - p.progress * 0.5;
        ctx.fill();
        ctx.globalAlpha = 1;
        alive.push(p);
    }
    particles = alive;
}
