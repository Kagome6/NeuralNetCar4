// simulation.js (Python主導アーキテクチャ対応版)

// --- HTML要素の取得 ---
const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');
const statusDisplay = document.getElementById('statusDisplay');
const connectButton = document.getElementById('connectButton');
const trainButton = document.getElementById('trainButton');
const stopTrainButton = document.getElementById('stopTrainButton');
const runButton = document.getElementById('runButton');
const stopRunButton = document.getElementById('stopRunButton');
const infoMode = document.getElementById('infoMode');
const infoEpisodeFrame = document.getElementById('infoEpisodeFrame');
const infoAgent0 = document.getElementById('infoAgent0');
const infoAgent1 = document.getElementById('infoAgent1');
const infoOther = document.getElementById('infoOther');
const progressBar = document.getElementById('trainProgress');
const sensorContainer = document.getElementById('sensorIndicators');
const actionContainer = document.getElementById('actionIndicators');

// --- シミュレーション定数 (描画に必要なもの、Python側と一部同期) ---
const SIM_WIDTH = 800;
const SIM_HEIGHT = 600;
// Agent
const AGENT_SIZE = 15.0;
const PURSUER_ID = 0;
const EVADER_ID = 1;
const EVADER_INITIAL_HEALTH = 10; // 体力バー計算用
const PURSUER_BULLET_COOLDOWN = 45; // クールダウン表示用
// Obstacle
const OBSTACLE_RADIUS = 7.0;
// Bullet
const BULLET_RADIUS = 3.0;
// Sensors (インジケーター用)
const NUM_OBSTACLE_SENSORS_A0 = 7; // Pursuerの障害物センサー数
const NUM_TARGET_SENSORS_A0 = 3;   // Pursuerのターゲットセンサー数
const SENSOR_RANGE_OBSTACLE_A0 = 130.0; // Pursuerの障害物センサー範囲
const SENSOR_RANGE_TARGET_A0 = SENSOR_RANGE_OBSTACLE_A0 * 1.5; // Pursuerのターゲットセンサー範囲
// Actions (インジケーター用)
const ACTION_LABELS_A0 = ['L', 'S', 'R', 'L+F', 'S+F', 'R+F']; // Pursuerのアクションラベル

// --- 描画スケール ---
const OBSTACLE_DRAW_SCALE = 0.8;
const AGENT_DRAW_SCALE = 1.2;
const BULLET_DRAW_SCALE = 1.0;

// Canvasサイズ設定
canvas.width = SIM_WIDTH;
canvas.height = SIM_HEIGHT;

// --- グローバル状態変数 ---
let agents = {};       // { agentId: Agentインスタンス }
let obstacles = [];    // [ {id, type, x, y, radius}, ... ]
let bullets = [];      // [ {id, type, owner_id, x, y, angle, radius, lifetime}, ... ]
let currentMode = 'idle';
let webSocket = null;
let serverAddress = "ws://localhost:8765"; // Pythonサーバーアドレス
let animationFrameId = null; // メインループID
let currentFrame = 0;      // サーバーから受信した最新フレーム番号
let currentEpisode = 0;    // サーバーから受信した最新エピソード番号
let lastTrainingStatus = {}; // 最後に受信した training_status データ

// --- Agent クラス (データ保持と描画のみ) ---
class Agent {
    constructor(id, type, x = 0, y = 0, angle = 0) {
        this.id = id;
        this.type = type;
        // --- サーバーから受信するデータ ---
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.is_active = true;
        this.collided_obstacle = false; // このフレームで衝突したか (描画用)
        this.collided_agent = false;    // このフレームで衝突したか (描画用)
        this.hit_by_bullet = false;     // このフレームで被弾したか (描画用, Evader)
        this.cooldown = 0;              // (Pursuer)
        this.health = (type === 'evader') ? EVADER_INITIAL_HEALTH : null; // (Evader)
        this.sensors_obstacle = [];     // 生のセンサー値 (インジケーター用)
        this.sensors_target = [];       // 生のセンサー値 (インジケーター用)
        this.lastAction = null;         // 最後に取ったアクション (インジケーター用)
        // --- ---
        this.size = AGENT_SIZE; // 描画用
    }

    // サーバーからのデータでエージェント状態を更新
    updateFromServerData(data) {
        if (!data) return;
        this.x = data.x ?? this.x;
        this.y = data.y ?? this.y;
        this.angle = data.angle ?? this.angle;
        this.is_active = data.is_active ?? this.is_active;
        // 衝突フラグは毎フレームサーバーから来るものを正とする
        this.collided_obstacle = data.collided_obstacle ?? false;
        this.collided_agent = data.collided_agent ?? false;
        this.hit_by_bullet = data.hit_by_bullet ?? false;

        if (this.type === 'pursuer') {
            this.cooldown = data.cooldown ?? this.cooldown;
        } else if (this.type === 'evader') {
            this.health = data.health ?? this.health;
        }
        // センサー値とアクションは存在する場合のみ更新 (デバッグ用)
        if (data.sensors_obstacle) this.sensors_obstacle = data.sensors_obstacle;
        if (data.sensors_target) this.sensors_target = data.sensors_target;
        // last_action は null の可能性もある
        this.lastAction = data.last_action;
    }

    draw(ctx) {
        // 衝突状態や体力に基づいて色を決定
        let fillColor = '#cccccc'; // Default inactive color
        if (this.is_active) {
             // アクティブでも衝突していたら色を変える
             if (this.collided_obstacle) {
                 fillColor = '#efb8c8'; // Pinkish for obstacle collision
             } else if (this.collided_agent) {
                 fillColor = '#d8bfd8'; // Light purple for agent collision
             } else if (this.type === 'evader' && this.hit_by_bullet) {
                 fillColor = '#ffaa88'; // Slightly darker orange for hit evader?
             } else {
                 // 通常のアクティブ色
                 fillColor = (this.type === 'pursuer') ? '#aaccff' : '#ffccaa'; // Blue: Pursuer, Orange: Evader
             }
        } else {
             // 非アクティブの理由で色分け (オプション)
             if (this.collided_obstacle) fillColor = '#c08080'; // Darker pink
             else if (this.collided_agent) fillColor = '#a080a0'; // Darker purple
             else if (this.type === 'evader' && this.health <= 0) fillColor = '#888888'; // Dark grey (health depleted)
             else fillColor = '#aaaaaa'; // Generic inactive
        }

        ctx.fillStyle = fillColor;
        const drawSize = this.size * AGENT_DRAW_SCALE;
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        // 三角形で描画
        ctx.beginPath();
        ctx.moveTo(drawSize * 0.7, 0);
        ctx.lineTo(-drawSize * 0.5, -drawSize * 0.5);
        ctx.lineTo(-drawSize * 0.5, drawSize * 0.5);
        ctx.closePath();
        ctx.fill();
        // ID表示
        ctx.fillStyle = 'white'; ctx.font = '10px Arial'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(this.id, 0, 0);
        ctx.restore();

        // クールダウンインジケーター描画 (Pursuer)
        if (this.type === 'pursuer' && this.cooldown > 0) {
             const cooldownRatio = Math.min(1.0, Math.max(0.0, this.cooldown / PURSUER_BULLET_COOLDOWN));
             const barWidth = this.size * 1.5;
             const barHeight = 4;
             const barX = this.x - barWidth / 2;
             const barY = this.y + this.size * 1.0; // エージェントの下
             ctx.fillStyle = 'rgba(100, 100, 255, 0.3)'; // 背景
             ctx.fillRect(barX, barY, barWidth, barHeight);
             ctx.fillStyle = 'rgba(0, 0, 200, 0.7)'; // ゲージ
             ctx.fillRect(barX, barY, barWidth * cooldownRatio, barHeight);
        }

        // 体力バー描画 (Evader)
        if (this.type === 'evader' && this.health !== null) {
             const healthRatio = Math.max(0, this.health / EVADER_INITIAL_HEALTH);
             const barWidth = this.size * 1.5;
             const barHeight = 4;
             const barX = this.x - barWidth / 2;
             const barY = this.y - this.size * 1.2; // エージェントの上
             ctx.fillStyle = '#555'; // 背景
             ctx.fillRect(barX, barY, barWidth, barHeight);
             ctx.fillStyle = (healthRatio > 0.5) ? '#00cc00' : (healthRatio > 0.2) ? '#cccc00' : '#cc0000'; // 緑/黄/赤
             ctx.fillRect(barX, barY, barWidth * healthRatio, barHeight);
        }
    }
}

// --- Geometry Helper (距離計算) ---
function distance(x1, y1, x2, y2) {
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
}

// --- 描画関数群 ---
function clearCanvas() {
    ctx.clearRect(0, 0, SIM_WIDTH, SIM_HEIGHT);
}

function drawBoundary() {
    ctx.strokeStyle = 'rgba(100, 100, 100, 0.6)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, SIM_WIDTH, SIM_HEIGHT);
}

function drawAgents() {
    Object.values(agents).forEach(agent => agent.draw(ctx));
}

function drawObstacles() {
    ctx.fillStyle = '#99cca5'; // Greenish
    obstacles.forEach(obs => {
        const drawRadius = obs.radius * OBSTACLE_DRAW_SCALE;
        drawArc(obs.x, obs.y, drawRadius);
        // Torus drawing (境界付近の描画)
        const drawThr = obs.radius * 2; // この距離より壁に近い場合に反対側にも描画
        if (obs.x < drawThr) drawArc(obs.x + SIM_WIDTH, obs.y, drawRadius);
        if (obs.x > SIM_WIDTH - drawThr) drawArc(obs.x - SIM_WIDTH, obs.y, drawRadius);
        if (obs.y < drawThr) drawArc(obs.x, obs.y + SIM_HEIGHT, drawRadius);
        if (obs.y > SIM_HEIGHT - drawThr) drawArc(obs.x, obs.y - SIM_HEIGHT, drawRadius);
        // Corners
        if (obs.x < drawThr && obs.y < drawThr) drawArc(obs.x + SIM_WIDTH, obs.y + SIM_HEIGHT, drawRadius);
        if (obs.x > SIM_WIDTH - drawThr && obs.y < drawThr) drawArc(obs.x - SIM_WIDTH, obs.y + SIM_HEIGHT, drawRadius);
        if (obs.x < drawThr && obs.y > SIM_HEIGHT - drawThr) drawArc(obs.x + SIM_WIDTH, obs.y - SIM_HEIGHT, drawRadius);
        if (obs.x > SIM_WIDTH - drawThr && obs.y > SIM_HEIGHT - drawThr) drawArc(obs.x - SIM_WIDTH, obs.y - SIM_HEIGHT, drawRadius);
    });
}

function drawBullets() {
    ctx.fillStyle = '#ffff99'; // Yellowish
    bullets.forEach(bullet => {
        const drawRadius = bullet.radius * BULLET_DRAW_SCALE;
        drawArc(bullet.x, bullet.y, drawRadius);
        // 弾はトーラス描画しない (Python側と合わせる)
    });
}

// 円を描画するヘルパー
function drawArc(x, y, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
}

// --- UI制御関数 ---
function updateStatusDisplay(status, message = '') {
    currentMode = status; // グローバル変数を更新
    statusDisplay.textContent = message || status.charAt(0).toUpperCase() + status.slice(1);
    statusDisplay.className = `status-indicator status-${status}`; // CSSクラスを更新

    const isConnected = (webSocket && webSocket.readyState === WebSocket.OPEN);
    connectButton.textContent = isConnected ? '切断' : '接続';
    connectButton.disabled = (status === 'connecting'); // 接続中は無効

    // 各モードに応じてボタンの有効/無効を切り替え
    trainButton.disabled = !(isConnected && status === 'idle');
    stopTrainButton.disabled = !(status === 'training');
    runButton.disabled = !(isConnected && status === 'idle');
    stopRunButton.disabled = !(status === 'running');

    // モードが変わったらプログレスバーをリセット
    updateProgress(0);
}

// 情報表示エリアを更新する関数
function updateInfoDisplay(data = {}) {
    // モード表示 (引数になければ現在のモードを使う)
    infoMode.textContent = data.mode ?? currentMode;

    // エピソード/フレーム表示 (サーバーからのデータ優先)
    const episodeStr = data.episode ?? currentEpisode ?? 'N/A';
    const frameStr = data.frame ?? currentFrame ?? 'N/A';
    infoEpisodeFrame.textContent = `Epi: ${episodeStr} / Frame: ${frameStr}`;

    // Agent 0 (Pursuer) 情報
    const rwd0 = data.rewards_current_ep?.[PURSUER_ID]?.toFixed(3) ?? data.rewards?.[PURSUER_ID]?.toFixed(3) ?? 'N/A';
    const eps0 = data.epsilons?.[PURSUER_ID]?.toFixed(3) ?? 'N/A';
    const loss0 = data.losses_last_batch?.[PURSUER_ID]?.toFixed(5) ?? 'N/A'; // 最新バッチ損失
    infoAgent0.textContent = `A0(追): Rwd ${rwd0}, Eps ${eps0}, Loss ${loss0}`;

    // Agent 1 (Evader) 情報
    const rwd1 = data.rewards_current_ep?.[EVADER_ID]?.toFixed(3) ?? data.rewards?.[EVADER_ID]?.toFixed(3) ?? 'N/A';
    const eps1 = data.epsilons?.[EVADER_ID]?.toFixed(3) ?? 'N/A';
    const loss1 = data.losses_last_batch?.[EVADER_ID]?.toFixed(5) ?? 'N/A';
    // 体力情報 (現在のエージェントデータから取得)
    const health1 = agents[EVADER_ID]?.health; // Optional chaining
    const health1Display = (health1 !== null && health1 !== undefined) ? `HP: ${health1}` : 'HP: N/A';
    infoAgent1.textContent = `A1(逃): Rwd ${rwd1}, Eps ${eps1}, Loss ${loss1}, ${health1Display}`;

    // その他のメッセージ表示
    if (data.message) {
        infoOther.textContent = data.message;
    } else if (currentMode === 'idle' || currentMode === 'error') {
        // アイドルやエラー時はクリア
         infoOther.textContent = '';
    }
    // episode_end メッセージは handleWebSocketMessage で直接セットする

    // 進捗バー更新 (Trainingモード時)
    if (currentMode === 'training' && data.progress !== undefined) {
        updateProgress(data.progress);
    }
}

// 学習進捗バーを更新
function updateProgress(progressPercent) {
    if (progressBar) {
        const value = Math.max(0, Math.min(100, progressPercent * 100)); // 0-100
        progressBar.value = value;
        // Training モードの時だけ表示
        progressBar.style.display = (currentMode === 'training') ? 'block' : 'none';
    }
}

// Agent 0 (Pursuer) のインジケーターを初期化
function initIndicators() {
    if (!sensorContainer || !actionContainer) return;
    sensorContainer.innerHTML = '';
    actionContainer.innerHTML = '';

    // センサーインジケーター (Pursuer用)
    const numTotalSensorsA0 = NUM_OBSTACLE_SENSORS_A0 + NUM_TARGET_SENSORS_A0;
    for (let i = 0; i < numTotalSensorsA0; i++) {
        const dot = document.createElement('div');
        dot.className = 'sensor-dot';
        dot.style.opacity = 0; // 初期は非表示
        // センサータイプで色分けするための data属性
        dot.dataset.type = (i < NUM_OBSTACLE_SENSORS_A0) ? 'obstacle' : 'target';
        sensorContainer.appendChild(dot);
    }

    // アクションインジケーター (Pursuer用)
    ACTION_LABELS_A0.forEach((label, i) => {
        const dot = document.createElement('div');
        // CSSクラス名に action-index を追加
        dot.className = `action-dot action-${i}`;
        actionContainer.appendChild(dot);
    });
}

// Agent 0 (Pursuer) のインジケーターを更新
function updateIndicators() {
    if (!sensorContainer || !actionContainer) return;

    const agent0 = agents[PURSUER_ID]; // Pursuerのインスタンスを取得

    // センサーインジケーター更新
    const sensorDots = sensorContainer.children;
    const expectedSensorCount = NUM_OBSTACLE_SENSORS_A0 + NUM_TARGET_SENSORS_A0;
    if (agent0 && agent0.is_active && sensorDots.length === expectedSensorCount) {
        // 障害物センサー
        if (agent0.sensors_obstacle && agent0.sensors_obstacle.length === NUM_OBSTACLE_SENSORS_A0) {
            agent0.sensors_obstacle.forEach((distanceVal, i) => {
                // センサー値を 0(遠い) ~ 1(近い) の強度に変換
                const intensity = 1.0 - Math.min(distanceVal / SENSOR_RANGE_OBSTACLE_A0, 1.0);
                if(sensorDots[i]) sensorDots[i].style.opacity = intensity.toFixed(2);
            });
        }
        // ターゲットセンサー
        if (agent0.sensors_target && agent0.sensors_target.length === NUM_TARGET_SENSORS_A0) {
            agent0.sensors_target.forEach((distanceVal, i) => {
                const intensity = 1.0 - Math.min(distanceVal / SENSOR_RANGE_TARGET_A0, 1.0);
                const targetDotIndex = NUM_OBSTACLE_SENSORS_A0 + i;
                if(sensorDots[targetDotIndex]) sensorDots[targetDotIndex].style.opacity = intensity.toFixed(2);
            });
        }
    } else {
        // エージェントが存在しないか非アクティブならインジケーターを非表示
        Array.from(sensorDots).forEach(dot => dot.style.opacity = 0);
    }

    // アクションインジケーター更新
    const actionDots = actionContainer.children;
    const lastAction = agent0 ? agent0.lastAction : null; // lastActionはnullの場合もある
    Array.from(actionDots).forEach((dot, idx) => {
        // lastAction (サーバーから受信) と一致するインデックスをアクティブにする
        dot.classList.toggle('active', idx === lastAction);
    });
}


// --- WebSocket関連 ---
function connectWebSocket() {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
         console.log("Already connected.");
         return;
    }
    updateStatusDisplay('connecting', '接続中...');
    webSocket = new WebSocket(serverAddress);

    webSocket.onopen = (event) => {
        console.log("WebSocket connection successful");
        // サーバーに接続成功を通知
        sendMessage({ type: 'client_hello', message: 'JS Frontend (Python-Driven) connected' });
        // ステータスは server_hello または mode_changed を待つ
    };

    webSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            // console.log("Message from server:", data.type); // デバッグ用
            handleWebSocketMessage(data);
        } catch (e) {
            console.error("Invalid JSON received:", event.data, e);
            updateStatusDisplay('error', '受信データエラー');
            // エラー時は切断した方が安全かも
            disconnectWebSocket();
        }
    };

    webSocket.onerror = (event) => {
        console.error("WebSocket error:", event);
        updateStatusDisplay('error', '接続エラー');
        webSocket = null;
        stopSimulationLoop(); // エラー時はループ停止
    };

    webSocket.onclose = (event) => {
        console.log("WebSocket disconnected:", event.code, event.reason);
        // サーバー側でエラーが発生して切断された場合、ステータスが 'error' のままになっていることがある
        if (currentMode !== 'error') {
             updateStatusDisplay('idle', '未接続');
        }
        webSocket = null;
        stopSimulationLoop(); // 切断時はループ停止
        // UIリセット
        clearSimulationState(); // エージェントなどの状態をクリア
        initIndicators(); // インジケーターリセット
        updateInfoDisplay({}); // 情報表示クリア
        clearCanvas(); drawBoundary(); // Canvasクリア
    };
}

function disconnectWebSocket() {
    if (webSocket) {
        console.log("Disconnecting WebSocket...");
        webSocket.close();
        // onclose イベントハンドラが残りの処理を行う
    }
}

function sendMessage(data) {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
        try {
            webSocket.send(JSON.stringify(data));
            console.log("Sent message:", data.type); // デバッグ用
        } catch (e) {
             console.error("Error sending message:", e);
        }
    } else {
        console.warn("WebSocket not connected. Cannot send message:", data);
    }
}

// WebSocketメッセージ処理
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'server_hello':
            console.log("Server says:", data.message);
            // サーバー接続確認、ステータスは mode_changed で更新されるのを待つ
            break;
        case 'init_state': // モード開始時の初期状態
            console.log("Received init_state");
            initializeSimulationState(data); // 状態を初期化
            // updateInfoDisplay(data); // フレーム情報などを更新
            if ((currentMode === 'running' || currentMode === 'training') && !animationFrameId) {
                 startSimulationLoop(); // ループ開始/再開
            }
            break;
        case 'update_state': // フレームごとの状態更新
            // console.log("Received update_state for frame:", data.frame); // デバッグ
            currentFrame = data.frame ?? currentFrame; // フレーム番号更新
            updateSimulationState(data); // 状態を更新
            // トレーニング中の情報表示は training_status で行うことが多いので、ここでは更新しない
            // if (currentMode === 'running') { updateInfoDisplay(data); }
            break;
        case 'training_status': // 学習中のステータス更新
            if (currentMode === 'training') {
                 lastTrainingStatus = data.status; // 最新ステータスを保持
                 currentEpisode = data.status.episode ?? currentEpisode; // エピソード番号更新
                 // ここで情報表示を更新
                 updateInfoDisplay(data.status);
            }
            break;
        case 'mode_changed': // サーバーのモード変更通知
            console.log(`Mode changed to: ${data.mode}, Message: ${data.message}`);
            const oldMode = currentMode;
            updateStatusDisplay(data.mode, data.message); // UI更新

            if ((data.mode === 'running' || data.mode === 'training')) {
                 // init_state 受信を待つので、ここではループを開始しない
                 // clearSimulationState(); // モード開始前にクリア
                 // initIndicators();
                 console.log(`Waiting for init_state to start ${data.mode} loop...`);
            } else { // idle or error になった場合
                 stopSimulationLoop(); // ループ停止
                 if (data.mode === 'idle') {
                     clearSimulationState(); // 状態クリア
                     initIndicators(); // インジケーターリセット
                     updateInfoDisplay({}); // 情報表示クリア
                     clearCanvas(); drawBoundary(); // Canvasクリア
                 }
            }
            break;
        case 'episode_end': // エピソード終了通知
            console.log("Received episode_end. Reason:", data.reason);
            // ループは止めずに継続 (Python側でリセット＆init_stateが送られてくる想定)
            if (infoOther) {
                // 終了理由を情報エリアに表示
                infoOther.textContent = `エピソード終了: ${data.reason}`;
            }
            // 必要ならここで一時停止などの処理を追加できるが、基本は自動継続
            break;
        case 'error': // サーバーからのエラー通知
            console.error("Server error:", data.message);
            updateStatusDisplay('error', `サーバーエラー: ${data.message}`);
            stopSimulationLoop(); // エラー時はループ停止
            disconnectWebSocket(); // エラー時は切断
            break;
        default:
            console.warn("Unknown message type received:", data.type, data);
    }
}

// --- シミュレーション状態更新関数 ---

// シミュレーション状態をクリアする
function clearSimulationState() {
    agents = {};
    obstacles = [];
    bullets = [];
    currentFrame = 0;
    currentEpisode = 0;
    lastTrainingStatus = {};
}

// init_state データでシミュレーション状態を初期化
function initializeSimulationState(data) {
    clearSimulationState(); // まず状態をクリア
    currentFrame = data.frame ?? 0;
    // Agentインスタンスを作成/更新
    if (data.agents) {
        data.agents.forEach(a_data => {
            agents[a_data.id] = new Agent(a_data.id, a_data.type);
            agents[a_data.id].updateFromServerData(a_data);
        });
    }
    obstacles = data.obstacles ?? [];
    bullets = data.bullets ?? [];
    // 初期描画はメインループに任せる
}

// update_state データでシミュレーション状態を更新
function updateSimulationState(data) {
    currentFrame = data.frame ?? currentFrame;
    // Agent状態を更新 (存在しない場合は警告を出すが、作成はしない)
    if (data.agents) {
        data.agents.forEach(a_data => {
            if (agents[a_data.id]) {
                agents[a_data.id].updateFromServerData(a_data);
            } else {
                console.warn(`Received update for non-existent agent ${a_data.id}. Ignored.`);
                 // 必要ならここで作成するロジックも入れられるが、init_stateで初期化されるはず
                 // agents[a_data.id] = new Agent(a_data.id, a_data.type);
                 // agents[a_data.id].updateFromServerData(a_data);
            }
        });
        // サーバーから来なくなったエージェントを削除（基本的には起こらないはず）
        // const receivedAgentIds = data.agents.map(a => a.id.toString());
        // for (const existingId in agents) {
        //     if (!receivedAgentIds.includes(existingId)) {
        //         console.warn(`Agent ${existingId} removed during update_state.`);
        //         delete agents[existingId];
        //     }
        // }
    }
    // 障害物と弾は常に最新リストで上書き
    obstacles = data.obstacles ?? obstacles;
    bullets = data.bullets ?? bullets;
}


// --- メインループ (描画専用) ---
function mainLoop() {
    // 1. Canvasクリア
    clearCanvas();

    // 2. 境界線描画
    drawBoundary();

    // 3. オブジェクト描画 (サーバーからの最新データに基づく)
    drawObstacles();
    drawBullets();
    drawAgents();

    // 4. UI更新 (インジケーター、必要なら情報表示も)
    updateIndicators(); // Agent 0 のセンサー/アクション表示
    // 情報表示エリアの更新 (モードに応じて)
    if(currentMode === 'running') {
        // 実行モードではフレーム情報と体力表示のみリアルタイム更新
        updateInfoDisplay({ frame: currentFrame });
    } else if (currentMode === 'training') {
        // 学習モードでは training_status 受信時に情報表示を更新するので、ここでは何もしないか、フレームのみ更新
        // updateInfoDisplay({ frame: currentFrame }); // フレームだけ更新する場合
    }


    // 5. 次のフレームを要求
    if (currentMode === 'training' || currentMode === 'running') {
        animationFrameId = requestAnimationFrame(mainLoop);
    } else {
        animationFrameId = null; // ループ停止
    }
}

// アニメーションループの開始
function startSimulationLoop() {
    if (!animationFrameId) {
        console.log("Starting simulation loop...");
        // ループ開始前に最初のフレームを描画（オプション）
        // clearCanvas(); drawBoundary(); drawObstacles(); drawBullets(); drawAgents(); updateIndicators();
        animationFrameId = requestAnimationFrame(mainLoop);
    }
}
// アニメーションループの停止
function stopSimulationLoop() {
    if (animationFrameId) {
        console.log("Stopping simulation loop...");
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --- 初期化とイベントリスナー ---
function initialize() {
    console.log("Initializing JS Frontend (Python-Driven)...");
    canvas.width = SIM_WIDTH;
    canvas.height = SIM_HEIGHT;

    initIndicators(); // インジケーター初期化
    updateStatusDisplay('idle', '未接続'); // 初期ステータス設定
    updateInfoDisplay({}); // 情報表示クリア
    clearSimulationState(); // 内部状態クリア
    clearCanvas();
    drawBoundary(); // 初期描画

    // --- Event Listeners ---
    connectButton.onclick = () => {
        if (webSocket && webSocket.readyState === WebSocket.OPEN) {
            disconnectWebSocket();
        } else {
            connectWebSocket();
        }
    };

    trainButton.onclick = () => {
        // ボタンが有効な時 (= idle & connected) のみ送信
        if (!trainButton.disabled) {
             sendMessage({ type: 'command', command: 'start_training' });
        } else { console.warn("Cannot start training. Check connection/mode."); }
    };

    stopTrainButton.onclick = () => {
        if (!stopTrainButton.disabled) {
            sendMessage({ type: 'command', command: 'stop_training' });
        }
    };

    runButton.onclick = () => {
        if (!runButton.disabled) {
            sendMessage({ type: 'command', command: 'start_running' });
        } else { console.warn("Cannot start running. Check connection/mode."); }
    };

    stopRunButton.onclick = () => {
         if (!stopRunButton.disabled) {
            sendMessage({ type: 'command', command: 'stop_running' });
            // サーバー側でモードが変わり、ループが止まるはずだが、即時停止させてもよい
            // stopSimulationLoop();
            // updateStatusDisplay('idle', '実行停止'); // UIだけ先に変える
        }
    };
}

// --- 初期化実行 ---
initialize();