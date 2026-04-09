import React, { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";

 ort.env.wasm.wasmPaths = {
    wasm: `${window.location}/ort/ort-wasm-simd-threaded.wasm`,
    mjs:`${window.location}/ort/ort-wasm-simd-threaded.mjs`,
};

const App = () => {
  // State for UI
  const [status, setStatus] = useState("Initializing...");
  const [userInput, setUserInput] = useState("");
  const [prediction, setPrediction] = useState<{
    label: string;
    score: number;
  } | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Refs to hold the session and vocab so they don't reload on every render
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const vocabRef = useRef<string[]>([]);

  const MODEL_PATH = "./ort/intent_model_tiny.ort";
  const VOCAB_PATH = "./ort/vocab.txt";
  const MAX_LENGTH = 32;


  // 1. Warm up the engine on load
  useEffect(() => {
    const init = async () => {
      try {
        setStatus("Loading Vocab...");
        const vocabResp = await fetch(VOCAB_PATH);
        const vocabText = await vocabResp.text();
        vocabRef.current = vocabText.split(/\r?\n/);

        setStatus("Loading Model (13MB)...");
        sessionRef.current = await ort.InferenceSession.create(MODEL_PATH);

        setIsReady(true);
        setStatus("Ready for input.");
      } catch (e) {
        setStatus(
          `Initialization Error: ${e instanceof Error ? e.message : "Check console"}`,
        );
      }
    };
    init();
  }, []);

  // 2. The Inference Logic
  const handleInference = async () => {
    if (!sessionRef.current || vocabRef.current.length === 0) return;

    setStatus("Processing...");

    try {
      // 1. Basic Tokenization Cleanup
      const tokens = userInput
        .toLowerCase()
        .replace(/[^\w\s]/g, "")
        .split(/\s+/)
        .filter((t) => t.length > 0);

      // 2. Map words to IDs using the vocab array
      const CLS_INDEX = vocabRef.current.indexOf("[CLS]"); // 101
      const SEP_INDEX = vocabRef.current.indexOf("[SEP]"); // 102
      const UNK_INDEX = vocabRef.current.indexOf("[UNK]"); // 100

      let inputIds = [CLS_INDEX];

      for (const token of tokens) {
        const id = vocabRef.current.indexOf(token);
        // If word not found, use [UNK] (100), NOT 101
        inputIds.push(id !== -1 ? id : UNK_INDEX);
      }

      inputIds.push(SEP_INDEX);

      // 3. Padding/Truncating to exactly 32
      while (inputIds.length < MAX_LENGTH) inputIds.push(0);
      const finalIds = inputIds.slice(0, MAX_LENGTH);
      // 4. Create ONNX Tensors
      const inputTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(finalIds.map(BigInt)),
        [1, 32],
      );
      const maskTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(finalIds.map((id) => (id !== 0 ? 1n : 0n))),
        [1, 32],
      );

      // 5. Run Model Inference
      const results = await sessionRef.current.run({
        input_ids: inputTensor,
        attention_mask: maskTensor,
      });

      const logits = results.logits.data as Float32Array;
      console.log("RAW LOGITS:", logits); // <--- ADD THIS

      // --- OPTION 2: SOFTMAX TEMPERATURE TRICK ---
      // Lower T (0.4 - 0.6) makes the results "sharper" and boosts confidence
      const temperature = 0.5;

      const expLogits = Array.from(logits).map((v) =>
        Math.exp(v / temperature),
      );
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probabilities = expLogits.map((v) => v / sumExp);

      // 6. Find the winner
      const maxProb = Math.max(...probabilities);
      const intentIndex = probabilities.indexOf(maxProb);

      // CRITICAL: Ensure these match the order in your config.json id2label!
      const labels = ["MOVE ON", "REPEAT", "CONFUSED", "FALLBACK"];

      setPrediction({
        label: labels[intentIndex],
        score: Math.round(maxProb * 100),
      });

      setStatus("Ready.");
    } catch (e) {
      setStatus("Inference Error.");
      console.error("Inference Error:", e);
    }
  };

  return (
    <div
      style={{
        padding: "40px",
        maxWidth: "600px",
        margin: "0 auto",
        fontFamily: "system-ui",
      }}
    >
      <h1>Local NLP Intent Test</h1>
      <p style={{ color: isReady ? "green" : "orange" }}>
        <strong>Status:</strong> {status}
      </p>

      <div style={{ display: "flex", gap: "10px", marginTop: "20px" }}>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="e.g. Can you play that again?"
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "4px",
            border: "1px solid #ccc",
          }}
          disabled={!isReady}
        />
        <button
          onClick={handleInference}
          disabled={!isReady || !userInput}
          style={{
            padding: "10px 20px",
            cursor: "pointer",
            background: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
          }}
        >
          Run Inference
        </button>
      </div>

      {prediction && (
        <div
          style={{
            marginTop: "30px",
            padding: "20px",
            backgroundColor: "#f0f4f8",
            borderRadius: "8px",
            borderLeft: "5px solid #007bff",
          }}
        >
          <h3 style={{ margin: 0 }}>
            Detected Intent:{" "}
            <span style={{ color: "#007bff" }}>{prediction.label}</span>
          </h3>
          <p
            style={{ margin: "10px 0 0 0", fontSize: "0.9rem", color: "#666" }}
          >
            Confidence: {prediction.score}%
          </p>
        </div>
      )}
    </div>
  );
};

export default App;
