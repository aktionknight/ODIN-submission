import React from "react";

export default function Logs({ log }) {
  return (
    <div className="mt-6 p-4 bg-gray-900 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-2">📜 ODIN Logs</h2>
      <pre className="bg-black p-2 rounded text-green-400 text-sm overflow-x-auto">
        {JSON.stringify(log, null, 2)}
      </pre>
    </div>
  );
}
