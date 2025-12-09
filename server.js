// server.js
// Webhook: Retell -> this server -> Snowflake Cortex Agent -> back to Retell

require("dotenv").config();
const express = require("express");
const bodyParser = require("body-parser");
const Retell = require("retell-sdk").default;

// Use global fetch if on Node 18+, otherwise lazy-load node-fetch
const fetch =
  global.fetch ||
  ((...args) =>
    import("node-fetch").then(({ default: f }) => f(...args)));

const {
  SNOWFLAKE_ACCOUNT_URL,
  SNOWFLAKE_PAT,
  SNOWFLAKE_AGENT_DATABASE,
  SNOWFLAKE_AGENT_SCHEMA,
  SNOWFLAKE_AGENT_NAME,
  RETELL_API_KEY,
  PORT,
} = process.env;

// Initialize Retell client for signature verification
const retellClient = new Retell({ apiKey: RETELL_API_KEY });

if (!SNOWFLAKE_ACCOUNT_URL || !SNOWFLAKE_PAT) {
  console.error("❌ Missing SNOWFLAKE_ACCOUNT_URL or SNOWFLAKE_PAT in .env");
  process.exit(1);
}

const app = express();
app.use(bodyParser.json());

/**
 * Helper: call Snowflake REST with PAT auth
 */
async function sfFetch(path, options = {}) {
  const url = `${SNOWFLAKE_ACCOUNT_URL}${path}`;

  const resp = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${SNOWFLAKE_PAT}`,
      "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
      ...(options.headers || {}),
    },
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(
      `Snowflake HTTP ${resp.status} ${resp.statusText}: ${text}`
    );
  }

  return resp;
}

/**
 * Create a Cortex thread for this request
 * POST /api/v2/cortex/threads
 */
async function createThread() {
  const resp = await sfFetch("/api/v2/cortex/threads", {
    method: "POST",
    body: JSON.stringify({
      origin_application: "SnowbearDemo",
    }),
  });
  const json = await resp.json();
  return json.id; // thread_id
}

/**
 * Run your DONUT_STORE_AGENT via Cortex Agents REST API
 * POST /api/v2/databases/{db}/schemas/{schema}/agents/{name}:run
 */
async function runSnowflakeAgent({ userText }) {
  const threadId = await createThread();

  const path = `/api/v2/databases/${encodeURIComponent(
    SNOWFLAKE_AGENT_DATABASE || "SNOWFLAKE_INTELLIGENCE"
  )}/schemas/${encodeURIComponent(
    SNOWFLAKE_AGENT_SCHEMA || "AGENTS"
  )}/agents/${encodeURIComponent(
    SNOWFLAKE_AGENT_NAME || "DONUT_STORE_AGENT"
  )}:run`;

  const body = {
    thread_id: threadId,
    parent_message_id: 0,
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: userText }],
      },
    ],
    tool_choice: { type: "auto" },
  };

  const resp = await sfFetch(path, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });

  const sseText = await resp.text();

  // Parse SSE to find the final "response" event
  let finalResponse = null;
  let lastEvent = null;

  for (const rawLine of sseText.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith("event:")) {
      lastEvent = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      if (lastEvent === "response") {
        const dataStr = line.slice("data:".length).trim();
        if (!dataStr) continue;
        finalResponse = JSON.parse(dataStr);
        break;
      }
    }
  }

  if (!finalResponse) {
    throw new Error("No final `response` event from Snowflake agent");
  }

  // Extract simple text content
  const textPart =
    finalResponse.content?.find((c) => c.type === "text")?.text ??
    JSON.stringify(finalResponse);

  return textPart;
}

/**
 * Retell Custom LLM WebSocket endpoint
 * Retell connects via WebSocket for real-time conversation
 */
const expressWs = require("express-ws")(app);

app.ws("/retell-llm", (ws, req) => {
  console.log("🔌 Retell WebSocket connected");

  let callId = null;

  ws.on("message", async (data) => {
    try {
      const message = JSON.parse(data);

      if (message.interaction_type === "call_details") {
        // Initial connection - store call details
        callId = message.call?.call_id;
        console.log(`📞 Call started: ${callId}`);

        // Send initial greeting
        const greeting = {
          response_id: 0,
          content: "Hi! I'm your data assistant. Ask me anything about your donut store data.",
          content_complete: true,
          end_call: false,
        };
        ws.send(JSON.stringify(greeting));
        return;
      }

      if (message.interaction_type === "response_required" || 
          message.interaction_type === "reminder_required") {
        const userText = message.transcript?.slice(-1)?.[0]?.content || "";
        const responseId = message.response_id;

        if (!userText) {
          ws.send(JSON.stringify({
            response_id: responseId,
            content: "I didn't catch that. Could you repeat your question?",
            content_complete: true,
            end_call: false,
          }));
          return;
        }

        console.log(`💬 User said: "${userText}"`);

        // Send "thinking" message immediately so user knows we're working
        ws.send(JSON.stringify({
          response_id: responseId,
          content: "Let me check that for you... ",
          content_complete: false,
          end_call: false,
        }));

        // Query Snowflake (this takes ~30 seconds)
        try {
          const answer = await runSnowflakeAgent({ userText });
          console.log(`✅ Snowflake response: ${answer.substring(0, 100)}...`);

          ws.send(JSON.stringify({
            response_id: responseId,
            content: answer,
            content_complete: true,
            end_call: false,
          }));
        } catch (err) {
          console.error("❌ Snowflake error:", err.message);
          ws.send(JSON.stringify({
            response_id: responseId,
            content: "Sorry, I had trouble getting that data. Could you try asking again?",
            content_complete: true,
            end_call: false,
          }));
        }
      }

      if (message.interaction_type === "hang_up") {
        console.log(`📴 Call ended: ${callId}`);
      }

    } catch (err) {
      console.error("❌ WebSocket error:", err);
    }
  });

  ws.on("close", () => {
    console.log("🔌 Retell WebSocket disconnected");
  });

  ws.on("error", (err) => {
    console.error("❌ WebSocket error:", err);
  });
});

/**
 * Health check endpoint
 */
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

/**
 * Legacy endpoint (keep for testing)
 */
app.post("/snowflake-agent", async (req, res) => {
  try {
    const userText =
      req.body.input || req.body.text || req.body.message || null;

    if (!userText) {
      return res.status(400).json({
        error: "Missing `input` (or `text` / `message`) in request body",
      });
    }

    const answer = await runSnowflakeAgent({ userText });
    res.json({ result: answer });
  } catch (err) {
    console.error("❌ Error in /snowflake-agent:", err);
    res.status(500).json({ error: err.message });
  }
});

const port = PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Snowbear webhook listening on port ${port}`);
  console.log(`🔗 Retell WebSocket endpoint: ws://localhost:${port}/retell-llm`);
});
