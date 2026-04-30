import type { DashboardState } from './dashboard';

declare const echarts: any;

let lossChart: any;
let accChart: any;

export function initCharts(state: DashboardState) {
    const lossEl = document.getElementById('loss-chart');
    const accEl = document.getElementById('acc-chart');
    if (!lossEl || !accEl) return;

    lossChart = echarts.init(lossEl, 'dark');
    accChart = echarts.init(accEl, 'dark');

    const baseOption = {
        backgroundColor: 'transparent',
        grid: { top: 65, right: 15, bottom: 35, left: 55 },
        tooltip: { 
            trigger: 'axis',
            confine: true,
            appendToBody: false,
        },
        toolbox: {
            feature: {
                saveAsImage: {
                    title: 'Save',
                    show: true,
                    type: 'png',
                    name: 'chart',
                    pixelRatio: 2,
                    backgroundColor: '#0d1117',
                },
                restore: { title: 'Reset', show: true },
            },
            right: 5,
            top: -5,
            itemSize: 12,
        },
        legend: { 
            top: 18, 
            textStyle: { color: '#8b949e', fontSize: 9 },
            type: 'scroll',
            itemWidth: 12,
            itemHeight: 8,
            itemGap: 5,
            orient: 'horizontal',
            height: 45,
        },
        xAxis: { 
            type: 'category', 
            name: 'Round', 
            nameLocation: 'middle',
            nameGap: 25,
            axisLine: { lineStyle: { color: '#30363d' } }, 
            axisLabel: { color: '#8b949e' },
        },
        dataZoom: [
            {
                type: 'inside',
                xAxisIndex: 0,
                start: 0,
                end: 100,
            },
            {
                type: 'slider',
                xAxisIndex: 0,
                start: 0,
                end: 100,
                height: 15,
                bottom: 0,
                borderColor: 'transparent',
                backgroundColor: 'rgba(48, 54, 61, 0.3)',
                fillerColor: 'rgba(88, 166, 255, 0.2)',
                handleStyle: {
                    color: '#58a6ff',
                    borderColor: '#58a6ff',
                },
                textStyle: { color: '#8b949e', fontSize: 9 },
                showDetail: false,
            },
        ],
    };

    lossChart.setOption({
        ...baseOption,
        title: { text: 'Loss', left: 'center', top: 0, textStyle: { color: '#c9d1d9', fontSize: 14 } },
        yAxis: { 
            type: 'value', 
            name: 'Loss', 
            nameLocation: 'middle',
            nameGap: 45,
            axisLine: { lineStyle: { color: '#30363d' }, show: true }, 
            axisLabel: { color: '#8b949e' }, 
            splitLine: { lineStyle: { color: '#21262d' } },
            nameTextStyle: { color: '#8b949e', fontSize: 12 },
        },
        series: [
            { name: 'Train Loss', type: 'line', data: [], smooth: true, lineStyle: { width: 2 }, itemStyle: { color: '#58a6ff' } },
            { name: 'Val Loss', type: 'line', data: [], smooth: true, lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#f0883e' } },
            { name: 'Test Loss', type: 'line', data: [], smooth: true, lineStyle: { width: 2, type: 'dotted' }, itemStyle: { color: '#3fb950' } },
        ],
    });

    accChart.setOption({
        ...baseOption,
        title: { text: 'Accuracy', left: 'center', top: 0, textStyle: { color: '#c9d1d9', fontSize: 14 } },
        yAxis: { 
            type: 'value', 
            name: 'Accuracy', 
            min: 0, 
            max: 1, 
            nameLocation: 'middle',
            nameGap: 45,
            axisLine: { lineStyle: { color: '#30363d' }, show: true }, 
            axisLabel: { color: '#8b949e' }, 
            splitLine: { lineStyle: { color: '#21262d' } },
            nameTextStyle: { color: '#8b949e', fontSize: 12 },
        },
        series: [
            { name: 'Train Acc', type: 'line', data: [], smooth: true, lineStyle: { width: 2 }, itemStyle: { color: '#58a6ff' } },
            { name: 'Val Acc', type: 'line', data: [], smooth: true, lineStyle: { width: 2, type: 'dashed' }, itemStyle: { color: '#f0883e' } },
            { name: 'Test Acc', type: 'line', data: [], smooth: true, lineStyle: { width: 2, type: 'dotted' }, itemStyle: { color: '#3fb950' } },
        ],
    });

    state.onUpdate.push(() => updateCharts(state));

    window.addEventListener('resize', () => {
        lossChart?.resize();
        accChart?.resize();
    });
}

const CLIENT_COLORS = ['#58a6ff', '#f0883e', '#a371f7', '#d2a8ff', '#79c0ff', '#ffa657'];

function alignDataToRounds(sourceRounds: number[], sourceData: (number | null)[], targetRounds: number[]): (number | null)[] {
    const result: (number | null)[] = [];
    const dataMap = new Map<number, number | null>();
    for (let i = 0; i < sourceRounds.length; i++) {
        dataMap.set(sourceRounds[i], sourceData[i]);
    }
    for (const round of targetRounds) {
        result.push(dataMap.has(round) ? dataMap.get(round)! : null);
    }
    return result;
}

function calculateAverage(perClient: Record<string, any>, metricKey: string, targetRounds: number[]): (number | null)[] {
    const clients = Object.keys(perClient);
    if (clients.length === 0) return targetRounds.map(() => null);

    return targetRounds.map((round, idx) => {
        let sum = 0;
        let count = 0;
        for (const cid of clients) {
            const pc = perClient[cid];
            const roundIdx = pc.rounds.indexOf(round);
            if (roundIdx !== -1) {
                const val = pc[metricKey][roundIdx];
                if (val !== null && val !== undefined) {
                    sum += val;
                    count++;
                }
            }
        }
        return count > 0 ? sum / count : null;
    });
}

// Store previous data to enable incremental updates
let prevLossData: any = null;
let prevAccData: any = null;

function buildSeriesData(
    name: string,
    data: (number | null)[],
    lineStyle: any,
    itemStyle: any,
    index: number
): any {
    return {
        name,
        type: 'line',
        data,
        smooth: true,
        lineStyle,
        itemStyle,
        // Enable incremental animation
        animationDuration: 300,
        animationDurationUpdate: 300,
        animationEasing: 'cubicOut',
        animationEasingUpdate: 'cubicOut',
        // Only animate the last point
        universalTransition: {
            enabled: true
        },
    };
}

function updateCharts(state: DashboardState) {
    const m = state.metrics;
    if (!m.rounds.length) return;

    const roundLabels = m.rounds.map(String);
    const isRingMode = state.topologyMode === 'ring';

    // Build loss chart series
    const lossSeries: any[] = [];
    const accSeries: any[] = [];

    if (isRingMode && Object.keys(m.per_client).length > 0) {
        // Ring mode: calculate average of per-client metrics
        const avgTrainLoss = calculateAverage(m.per_client, 'train_loss', m.rounds);
        const avgValLoss = calculateAverage(m.per_client, 'val_loss', m.rounds);
        const avgTrainAcc = calculateAverage(m.per_client, 'train_acc', m.rounds);
        const avgValAcc = calculateAverage(m.per_client, 'val_acc', m.rounds);

        lossSeries.push(buildSeriesData(
            'Train Loss (Avg)',
            avgTrainLoss,
            { width: 2, color: '#58a6ff' },
            { color: '#58a6ff' },
            0
        ));

        lossSeries.push(buildSeriesData(
            'Val Loss (Avg)',
            avgValLoss,
            { width: 2, type: 'dashed', color: '#f0883e' },
            { color: '#f0883e' },
            1
        ));

        lossSeries.push(buildSeriesData(
            'Test Loss (Global)',
            m.test_loss,
            { width: 2, type: 'dotted', color: '#3fb950' },
            { color: '#3fb950' },
            2
        ));

        accSeries.push(buildSeriesData(
            'Train Acc (Avg)',
            avgTrainAcc,
            { width: 2, color: '#58a6ff' },
            { color: '#58a6ff' },
            0
        ));

        accSeries.push(buildSeriesData(
            'Val Acc (Avg)',
            avgValAcc,
            { width: 2, type: 'dashed', color: '#f0883e' },
            { color: '#f0883e' },
            1
        ));

        accSeries.push(buildSeriesData(
            'Test Acc (Global)',
            m.test_acc,
            { width: 2, type: 'dotted', color: '#3fb950' },
            { color: '#3fb950' },
            2
        ));
    } else {
        // Client-server mode: show global metrics only
        lossSeries.push(buildSeriesData(
            'Train Loss',
            m.train_loss,
            { width: 2 },
            { color: '#58a6ff' },
            0
        ));
        lossSeries.push(buildSeriesData(
            'Val Loss',
            m.val_loss,
            { width: 2, type: 'dashed' },
            { color: '#f0883e' },
            1
        ));
        lossSeries.push(buildSeriesData(
            'Test Loss',
            m.test_loss,
            { width: 2, type: 'dotted' },
            { color: '#3fb950' },
            2
        ));

        accSeries.push(buildSeriesData(
            'Train Acc',
            m.train_acc,
            { width: 2 },
            { color: '#58a6ff' },
            0
        ));
        accSeries.push(buildSeriesData(
            'Val Acc',
            m.val_acc,
            { width: 2, type: 'dashed' },
            { color: '#f0883e' },
            1
        ));
        accSeries.push(buildSeriesData(
            'Test Acc',
            m.test_acc,
            { width: 2, type: 'dotted' },
            { color: '#3fb950' },
            2
        ));
    }

    // Use notMerge: false to enable incremental updates
    // This allows ECharts to animate only the changed data
    const updateOption = {
        notMerge: false,
        lazyUpdate: false,
        silent: true,
    };

    lossChart?.setOption({
        xAxis: { data: roundLabels },
        series: lossSeries,
        // Add global animation settings
        animation: true,
        animationDuration: 300,
        animationDurationUpdate: 300,
        animationEasing: 'cubicOut',
        animationEasingUpdate: 'cubicOut',
    }, updateOption);

    accChart?.setOption({
        xAxis: { data: roundLabels },
        series: accSeries,
        animation: true,
        animationDuration: 300,
        animationDurationUpdate: 300,
        animationEasing: 'cubicOut',
        animationEasingUpdate: 'cubicOut',
    }, updateOption);

    // Store current data for next comparison
    prevLossData = { rounds: [...m.rounds], series: lossSeries };
    prevAccData = { rounds: [...m.rounds], series: accSeries };
}
