import React from "react";

export default function Controls() {
  return (
    <div className="mt-6 p-4 bg-gray-900 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-2">⚙️ Mission Parameters</h2>
      <div className="flex gap-4">
        <button className="px-4 py-2 bg-blue-600 rounded">Simulate Hazard</button>
        <button className="px-4 py-2 bg-green-600 rounded">Recompute</button>
      </div>
    </div>
  );
}
