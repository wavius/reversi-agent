let Module;
let engine;
let game;
let session;
let boardEl = document.getElementById("board");
let messageEl = document.getElementById("message");
let opponentSelect = document.getElementById("opponent");
let isPlayerTurn = true;
let isAutoPlay = false;
let autoAgentColor = 2; // Default to white
const COLOR_BLACK = 1;
const COLOR_WHITE = 2;

async function init() {
    messageEl.textContent = "Loading...";

    // load wasm
    Module = await createReversiModule();
    engine = new Module.BatchedMCTSEngine(1.0);

    // load onnx model (only needed for agent opponent)
    try {
        session = await ort.InferenceSession.create('reversi_model.onnx', { executionProviders: ['webgl', 'wasm'] });
    } catch (e) {
        console.warn("ONNX model failed to load, Agent opponent disabled:", e);
        session = null;
    }

    startNewGame();
}

function startNewGame() {
    isAutoPlay = false;
    if (game) game.delete();
    game = new Module.Game();
    engine.clear_cache();
    isPlayerTurn = true;
    updateUI();
}

function startAutoPlay() {
    isAutoPlay = true;
    autoAgentColor = (Math.random() < 0.5) ? COLOR_BLACK : COLOR_WHITE;
    if (game) game.delete();
    game = new Module.Game();
    engine.clear_cache();
    isPlayerTurn = false;
    updateUI();
}

function updateUI() {
    boardEl.innerHTML = "";
    let board = game.get_board();
    let current_player = game.get_current_player();
    let currentValue = current_player.value;
    let is_over = game.is_game_over();

    for (let i = 0; i < 64; i++) {
        let cell = document.createElement("div");
        cell.className = "cell";

        let pieceColor = board.get_piece(Math.floor(i / 8), i % 8);
        if (pieceColor.value !== 0) {
            let piece = document.createElement("div");
            piece.className = "piece show " + (pieceColor.value === COLOR_BLACK ? "black" : "white");
            cell.appendChild(piece);
        } else if (!is_over && currentValue === parseInt(document.getElementById("player-color").value) && !isAutoPlay && board.is_valid_move(Math.floor(i / 8), i % 8, current_player)) {
            cell.classList.add("valid-move");
            cell.onclick = () => handlePlayerMove(i);
        }
        boardEl.appendChild(cell);
    }
    board.delete();

    if (is_over) {
        let winner = game.get_winner();
        if (winner.value === COLOR_BLACK) {
            messageEl.textContent = isAutoPlay ? "Minimax (Black) wins!" : "You win 😼";
        } else if (winner.value === COLOR_WHITE) {
            messageEl.textContent = isAutoPlay ? "Agent (White) wins!" : "You lose 😿";
        } else {
            messageEl.textContent = "Draw!";
        }
    } else {
        let playerColor = parseInt(document.getElementById("player-color").value);
        let isHumanTurn = (currentValue === playerColor);
        
        if (isAutoPlay) {
            isPlayerTurn = false;
            let isAgentTurn = (currentValue === autoAgentColor);
            messageEl.textContent = "Auto playing... " + (isAgentTurn ? "(Agent)" : "(Minimax)");
            setTimeout(() => makeAIMove(isAgentTurn ? "agent" : "minimax"), 50);
        } else if (!isHumanTurn) {
            isPlayerTurn = false;
            messageEl.textContent = "Thinking...";
            setTimeout(() => makeAIMove(opponentSelect.value), 50);
        } else {
            isPlayerTurn = true;
            messageEl.textContent = "Your turn!";
        }
    }
}

function handlePlayerMove(pos) {
    if (!isPlayerTurn) return;
    game.apply_move_fast(Math.floor(pos / 8), pos % 8);
    updateUI();
}

async function makeAIMove(alg) {
    let board = game.get_board();
    let current_player = game.get_current_player();
    let move = -1;

    let totalPieces = board.count_black_pieces() + board.count_white_pieces();
    
    if (isAutoPlay && totalPieces < 8) {
        let valid_moves = [];
        for (let i = 0; i < 64; i++) {
            if (board.is_valid_move(Math.floor(i / 8), i % 8, current_player)) {
                valid_moves.push(i);
            }
        }
        if (valid_moves.length > 0) {
            move = valid_moves[Math.floor(Math.random() * valid_moves.length)];
        }
    } else if (alg === "greedy") {
        move = Module.greedy_move(board, current_player);
    } else if (alg === "minimax") {
        move = Module.minimax_move(board, current_player);
    } else {
        if (!session) {
            messageEl.textContent = "Agent unavailable (ONNX failed to load).";
            board.delete();
            current_player.delete();
            return;
        }
        // mcts loop
        let games_vec = new Module.VectorGame();
        games_vec.push_back(game);
        engine.initialize(games_vec);

        let val = parseInt(document.getElementById("mcts-sims").value);
        let mctsSims = isNaN(val) ? 1000 : Math.min(1000, Math.max(1, val));
        document.getElementById("mcts-sims").value = mctsSims;
        for (let sim = 0; sim < mctsSims; sim++) {
            let req = engine.prepare_evaluation();
            if (req.p0.size() > 0) {
                // decode bitboards and run onnx inference
                let p0 = BigInt(req.p0.get(0));
                let p1 = BigInt(req.p1.get(0));

                let tensorData = new Float32Array(128);
                for (let i = 0; i < 64; i++) {
                    if ((p0 & (1n << BigInt(i))) !== 0n) tensorData[i] = 1.0;
                    if ((p1 & (1n << BigInt(i))) !== 0n) tensorData[64 + i] = 1.0;
                }

                let inputTensor = new ort.Tensor('float32', tensorData, [1, 2, 8, 8]);
                let results = await session.run({ "input": inputTensor });

                let policyArray = results.policy.data;
                let valueArray = results.value.data;

                let policiesVec = new Module.VectorVectorFloat();
                let singlePolicyVec = new Module.VectorFloat();
                for (let i = 0; i < 64; i++) singlePolicyVec.push_back(policyArray[i]);
                policiesVec.push_back(singlePolicyVec);

                let valuesVec = new Module.VectorFloat();
                valuesVec.push_back(valueArray[0]);

                engine.process_evaluation(policiesVec, valuesVec);

                singlePolicyVec.delete();
                policiesVec.delete();
                valuesVec.delete();
            } else {
                let emptyP = new Module.VectorVectorFloat();
                let emptyV = new Module.VectorFloat();
                engine.process_evaluation(emptyP, emptyV);
                emptyP.delete();
                emptyV.delete();
            }
        }

        let batch_probs = engine.get_action_probs(0.0);
        let probs = batch_probs.get(0);
        let max_prob = -1;
        for (let i = 0; i < 64; i++) {
            if (probs.get(i) > max_prob) {
                max_prob = probs.get(i);
                move = i;
            }
        }
        batch_probs.delete();
        games_vec.delete();
    }

    board.delete();

    if (move !== -1) {
        game.apply_move_fast(Math.floor(move / 8), move % 8);
    }
    updateUI();
}

document.getElementById("reset-btn").onclick = startNewGame;
document.getElementById("auto-btn").onclick = startAutoPlay;
document.getElementById("player-color").onchange = startNewGame;
document.getElementById("opponent").onchange = startNewGame;

window.onload = init;
