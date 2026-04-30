import { initDashboard } from './dashboard';
import { initTopology } from './topology';
import { initCharts } from './charts';
import { initClientTable } from './client-table';
import { initEventLog } from './event-log';

document.addEventListener('DOMContentLoaded', () => {
    const state = initDashboard();
    initTopology(state);
    initCharts(state);
    initClientTable(state);
    initEventLog(state);
});
