import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const PYTHON = path.join(PROJECT_ROOT, "myenv", "bin", "python3");

const PIPELINE_STEPS: Record<string, { script: string; args?: string[]; label: string }> = {
  download: { script: "download_recordings.py", label: "Download Recordings" },
  cluster: { script: "cluster_demographics.py", label: "Cluster Demographics" },
  training: { script: "build_training_data.py", label: "Build Training Data" },
  finetune: { script: "fine_tune.py", label: "Fine-Tune Models" },
  agents: { script: "run_agents.py", args: ["--max-steps", "15", "--max-duration", "120"], label: "Run Agents" },
};

export async function POST(req: NextRequest) {
  const { step } = await req.json();
  const config = PIPELINE_STEPS[step];

  if (!config) {
    return new Response(JSON.stringify({ error: `Unknown step: ${step}` }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;

      const send = (type: string, data: unknown) => {
        if (closed) return;
        try {
          const payload: Record<string, unknown> = { type };
          if (typeof data === "string") {
            payload.message = data;
          } else if (typeof data === "object" && data !== null) {
            Object.assign(payload, data);
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
        } catch {
          closed = true;
        }
      };

      const closeStream = () => {
        if (closed) return;
        closed = true;
        try { controller.close(); } catch { /* already closed */ }
      };

      send("status", { status: "running", step, label: config.label });

      const args = [path.join(PROJECT_ROOT, config.script), ...(config.args || [])];
      const proc = spawn(PYTHON, args, {
        cwd: PROJECT_ROOT,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      let lineBuffer = "";

      const processLine = (line: string) => {
        if (!line.trim() || closed) return;

        if (line.includes("ERROR:") || line.includes("Error:")) {
          send("error", line.trim());
        } else if (line.includes("WARNING:") || line.includes("Warning:")) {
          send("warning", line.trim());
        } else if (line.match(/^\s*\d+\/\d+/) || line.includes("...")) {
          send("progress", line.trim());
        } else if (line.startsWith("===") || line.startsWith("---")) {
          send("section", line.trim());
        } else {
          send("log", line.trim());
        }
      };

      const handleData = (data: Buffer) => {
        if (closed) return;
        lineBuffer += data.toString();
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() || "";
        for (const line of lines) {
          processLine(line);
        }
      };

      proc.stdout.on("data", handleData);
      proc.stderr.on("data", handleData);

      proc.on("close", (code) => {
        if (lineBuffer.trim()) processLine(lineBuffer);
        send("done", { code, step, success: code === 0 });
        closeStream();
      });

      proc.on("error", (err) => {
        send("error", `Process error: ${err.message}`);
        send("done", { code: 1, step, success: false });
        closeStream();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
