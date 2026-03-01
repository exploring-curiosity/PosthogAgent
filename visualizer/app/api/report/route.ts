import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const REPORT_PATH = path.join(PROJECT_ROOT, "data", "reports", "comparative_report_latest.json");

export async function GET() {
  try {
    if (!fs.existsSync(REPORT_PATH)) {
      return new Response(JSON.stringify({ error: "No report found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    const data = fs.readFileSync(REPORT_PATH, "utf-8");
    return new Response(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
