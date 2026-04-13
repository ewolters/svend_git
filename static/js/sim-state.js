let currentSimId = null;
let layout = { stations: [], connections: [], sources: [], sinks: [], work_centers: [], operators: [] };
let selectedElement = null;
let currentTool = 'select';
let connectFrom = null;

// Canvas transform state
let canvasZoom = 1;
let canvasPanX = 0;
let canvasPanY = 0;
let isPanning = false;
let panStart = { x: 0, y: 0 };
let dragElement = null;
let dragOffset = { x: 0, y: 0 };

// DES engine instance
let des = null;
let animFrameId = null;

// Saved runs for comparison
let savedRuns = [];

// Element ID counter
let nextId = 1;
function genId(prefix) { return `${prefix}-${nextId++}`; }

// Transport type definitions — speed, constraints, failure modes
const TRANSPORT_TYPES = {
    none:       { label: 'None (instant)', speed: Infinity, needsOperator: false, breakdownRate: 0, icon: '—' },
    walk:       { label: 'Walk (small parts)', speed: 1.0,  needsOperator: false, breakdownRate: 0, icon: 'M12 4a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM12 10v5M9 12h6M10 19l2-4 2 4', maxContainerKg: 15 },
    hand_cart:  { label: 'Hand Cart', speed: 1.2,           needsOperator: false, breakdownRate: 0, icon: 'M3 7h14l1.5 9H4.5zM7 19a2 2 0 1 0 4 0 2 2 0 0 0-4 0zM13 19a2 2 0 1 0 4 0 2 2 0 0 0-4 0z', maxContainerKg: 100 },
    pallet_jack:{ label: 'Manual Pallet Jack', speed: 1.5,  needsOperator: false, breakdownRate: 0, icon: 'M4 8h16M4 12h16M7 8v4M13 8v4M10 4v4M10 12v4', maxContainerKg: 2000 },
    electric_pj:{ label: 'Electric Pallet Jack', speed: 2.5,needsOperator: true,  breakdownRate: 0.003, icon: 'M13 2L3 14h9l-1 8 10-12h-9l1-8z', maxContainerKg: 3000 },
    forklift:   { label: 'Forklift', speed: 3.0,            needsOperator: true,  breakdownRate: 0.008, icon: 'M3 16V6h5v10M8 10h4V6M1 16h14M4 19a2 2 0 1 0 4 0 2 2 0 0 0-4 0zM10 19a2 2 0 1 0 4 0 2 2 0 0 0-4 0z', maxContainerKg: 5000 },
    agv:        { label: 'AGV (automated)', speed: 2.0,     needsOperator: false, breakdownRate: 0.005, icon: 'M4 6h16v12H4zM9 9h6v6H9zM9 3v3M15 3v3M3 10h1M3 14h1M20 10h1M20 14h1', maxContainerKg: 1500 },
};

// MinHeap provided by svend-sim-core.js
