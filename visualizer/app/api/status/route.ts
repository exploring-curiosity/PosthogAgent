import { NextRequest } from "next/server";
import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const DATA_DIR = path.join(PROJECT_ROOT, "data");

function dirCount(dir: string): number {
  try {
    return fs.readdirSync(dir).filter((f) => !f.startsWith(".")).length;
  } catch {
    return 0;
  }
}

function readJSON(filePath: string): unknown {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

export async function GET(_req: NextRequest) {
  const recordings = dirCount(path.join(DATA_DIR, "recordings"));
  const parsed = dirCount(path.join(DATA_DIR, "parsed"));
  const descriptions = dirCount(path.join(DATA_DIR, "descriptions"));
  const embeddings = dirCount(path.join(DATA_DIR, "embeddings"));
  const trainingFiles = dirCount(path.join(DATA_DIR, "training"));
  const modelFiles = dirCount(path.join(DATA_DIR, "models"));

  const clusters = readJSON(path.join(DATA_DIR, "clusters", "clusters.json")) as Record<string, unknown> | null;
  const models = readJSON(path.join(DATA_DIR, "models", "models.json")) as Record<string, unknown> | null;
  const report = readJSON(path.join(DATA_DIR, "reports", "comparative_report_latest.json")) as Record<string, unknown> | null;

  // Determine which steps are complete
  const steps = {
    download: { done: recordings > 0, count: recordings },
    cluster: { done: clusters !== null, count: clusters ? (clusters as Record<string, unknown>).num_clusters || 0 : 0, data: clusters },
    training: { done: trainingFiles > 0, count: trainingFiles },
    finetune: { done: models !== null, data: models },
    agents: { done: report !== null, data: report },
  };

  return Response.json({
    steps,
    counts: { recordings, parsed, descriptions, embeddings, trainingFiles, modelFiles },
    hasReport: report !== null,
  });
}
